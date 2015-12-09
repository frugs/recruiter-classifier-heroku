import email
import base64
import configparser
import argparse
import httplib2
import apiclient
import oauth2client.client
import oauth2client.tools
import oauth2client.file
import pytdc.data
import pytdc.classification
import pydl
import gensim.models


def main():
    config = configparser.ConfigParser()
    config.read("default.conf")
    config.read("user.conf")

    client_secrets_path = config["Paths"]["client_secrets_path"]
    credential_store_path = config["Paths"]["user_credentials"]
    classification_network_path = config["Paths"]["classification_network_path"]
    word_vector_model_path = config["Paths"]["word_vector_model_path"]

    classified_label_name = config["Email"]["classified_label_name"]
    recruiter_label_name = config["Email"]["recruiter_label_name"]
    email_classification_filter = config["Email"]["email_classification_filter"]

    service = create_gmail_api_service(client_secrets_path, credential_store_path)

    network = pydl.Network([])
    network.load(classification_network_path)
    model = gensim.models.Word2Vec.load_word2vec_format(word_vector_model_path, binary=True)

    classified_label_id, recruiter_label_id = initialise_labels(service, classified_label_name, recruiter_label_name)

    unread_inbox_message_ids = get_unread_message_ids_from_inbox(service,
                                                                 classified_label_name,
                                                                 email_classification_filter)
    recruiter_message_ids = [message_id
                             for message_id
                             in unread_inbox_message_ids
                             if is_message_from_recruiter(network, model, service, message_id)]

    label_messages(service, classified_label_id, recruiter_label_id, unread_inbox_message_ids, recruiter_message_ids)

    print("Checked " + str(len(unread_inbox_message_ids)) + " messages, " +
          "marked " + str(len(recruiter_message_ids)) + " as recruiter message.")


def create_gmail_api_service(client_secrets_path, credential_store_path):
    scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    flags = argparse.ArgumentParser(parents=[oauth2client.tools.argparser]).parse_args()

    storage = oauth2client.file.Storage(credential_store_path)
    flow = oauth2client.client.flow_from_clientsecrets(client_secrets_path, scopes)
    credentials = storage.get() if storage.get() is not None else oauth2client.tools.run_flow(flow, storage, flags)
    return apiclient.discovery.build("gmail", "v1", http=credentials.authorize(httplib2.Http()))


def initialise_labels(service, classified_label_name, recruiter_label_name):
    labels = service.users().labels().list(userId="me").execute()["labels"]

    classified_label_id = next((label["id"] for label in labels if label["name"] == classified_label_name), "")
    if not classified_label_id:
        body = {"messageListVisibility": "show", "name": classified_label_name, "labelListVisibility": "labelShow"}
        response = service.users().labels().create(userId="me",body=body).execute()
        classified_label_id = response["id"]

    recruiter_label_id = next((label["id"] for label in labels if label["name"] == recruiter_label_name), "")
    if not recruiter_label_id:
        body = {"messageListVisibility": "hide", "name": recruiter_label_name, "labelListVisibility": "labelShow"}
        response = service.users().labels().create(userId="me",body=body).execute()
        recruiter_label_id = response["id"]

    return classified_label_id, recruiter_label_id


def get_unread_message_ids_from_inbox(service, classified_label_name, email_classification_filter):
    query = "-label:" + classified_label_name + " " + email_classification_filter
    results = service.users().messages().list(userId="me", labelIds=["UNREAD", "INBOX"], q=query).execute()
    return [message_key.get("id") for message_key in results.get("messages", [])]


def is_message_from_recruiter(network, model, service, message_id):
    get_message_response = service.users().messages().get(userId="me", id=message_id, format="raw").execute()
    message = email.message_from_bytes(base64.urlsafe_b64decode(get_message_response["raw"]))
    message_contents = pytdc.data.words_from_email(message)
    input_vector = pytdc.data.vectorise_words_using_word_vector_model(message_contents, model, 200)
    return pytdc.classification.classify_input(network, input_vector, lambda x: x[0, 0] - x[1, 0] > 0.47)


def label_messages(service, classified_label_id, recruiter_label_id, unread_inbox_message_ids, recruiter_message_ids):
    for message_id in unread_inbox_message_ids:
        add_label_ids = [classified_label_id,
                         recruiter_label_id] if message_id in recruiter_message_ids else [classified_label_id]
        remove_label_ids = ["INBOX"] if message_id in recruiter_message_ids else []

        body = {"addLabelIds": add_label_ids, "removeLabelIds": remove_label_ids}
        service.users().messages().modify(userId="me", id=message_id, body=body).execute()

if __name__ == "__main__":
    main()
