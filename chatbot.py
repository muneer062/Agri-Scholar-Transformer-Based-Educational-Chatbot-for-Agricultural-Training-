import torch                                   # For running the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from joblib import load                        # For loading the label encoder

# Model configuration (must match training)
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64

# Paths to saved model and label encoder
BEST_MODEL_PATH = "models/best_model.pt"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load label encoder
label_encoder = load(LABEL_ENCODER_PATH)
print("Loaded label encoder with classes:", label_encoder.classes_)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Number of labels
num_labels = len(label_encoder.classes_)

# Load model architecture
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# Load trained weights
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

# Move model to device
model.to(device)
model.eval()   # Evaluation mode


# Predefined responses for each intent
intent_responses = {
    "fertilizer_wheat": "For wheat, follow local recommendations for NPK dose per acre and split applications.",
    "pest_control_rice": "For rice pests, inspect fields regularly and use recommended pesticides with proper dose and timing.",
    "irrigation_maize": "Maize needs irrigation at sowing, knee-high stage, tasseling, and grain filling for best yield.",
    "fertilizer_rice": "Use balanced NPK for rice based on soil test; avoid overuse of nitrogen to prevent lodging.",
    "disease_control_wheat": "Identify the wheat disease (e.g., rust, blight) and apply recommended fungicides at the right stage.",
    "irrigation_vegetables": "Vegetables need frequent light irrigations; avoid waterlogging and adjust based on soil moisture.",
    "pest_control_cotton": "Monitor cotton for bollworms and sucking pests; use integrated pest management and safe pesticides.",
    "irrigation_wheat": "Critical irrigations for wheat are at crown root initiation, tillering, and grain filling stages.",
    "pest_control_maize": "For maize pests, use seed treatment and targeted sprays when pest population crosses threshold."
    # You can add new intents and responses as you expand your dataset
}


def predict_intent(question_text):
    """
    Given a user question (string), return the predicted intent label (string).
    """

    # Tokenize and encode the input question
    encoding = tokenizer.encode_plus(
        question_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Disable gradient computation
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        # Get index of highest-scoring class
        _, predicted_class_id = torch.max(logits, dim=1)

    # Convert tensor to Python int
    predicted_class_id = predicted_class_id.cpu().item()

    # Convert back to label string
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]

    return predicted_label


def get_response_for_intent(intent_label):
    """
    Map an intent label to a predefined response text.
    """
    # If label is known, return its response; otherwise, a default
    return intent_responses.get(
        intent_label,
        "I am not sure about that yet. Please consult a local agricultural expert."
    )


def main():
    """
    Run a simple command-line chatbot loop.
    """
    print("Welcome to AgriEduBot! Ask a question about crops or farming.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ")

        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q"]:
            print("AgriEduBot: Goodbye! Happy farming.")
            break

        # Predict intent
        predicted_intent = predict_intent(user_input)

        # Get response text
        bot_response = get_response_for_intent(predicted_intent)

        # Show predicted intent (for debugging / educational purpose)
        print(f"(Predicted intent: {predicted_intent})")

        # Show bot response
        print("AgriEduBot:", bot_response)
        print()  # Blank line for readability


if __name__ == "__main__":
    main()