from flask import Flask, request, render_template
import openai

app = Flask(__name__)

# Placeholder function for your RAG pipeline
def run_rag_pipeline(query):
    # Mock function to simulate pipeline response
    # Replace this with your actual RAG pipeline logic
    return f"Response for '{query}' from the RAG pipeline."

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            response = run_rag_pipeline(query)  # Call your RAG pipeline here

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
