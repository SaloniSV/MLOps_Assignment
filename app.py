from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email,load_classes, save_classes
app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200





@app.route("/api/v1/add-class", methods=['POST'])
def add_class():
    if request.is_json:
        data = request.get_json()  # Get JSON data from the request body
        new_class = data.get('class')

        if new_class:
            # Load existing classes
            classes = load_classes()

            # Check if the class is already in the list
            if new_class in classes:
                return jsonify({"message": f"Class '{new_class}' already exists!"}), 400
            else:
                # Add the new class and save it
                classes.append(new_class)
                save_classes(classes)
                
                # Reload the updated classes into EMAIL_CLASSES
                global EMAIL_CLASSES
                EMAIL_CLASSES = load_classes()  # Reload classes from the file

                return jsonify({"message": f"Class '{new_class}' added successfully!"}), 200
        else:
            return jsonify({"error": "No class provided in request body"}), 400
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
