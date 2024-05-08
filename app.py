from flask import Flask, request, Response
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

PROMPT = """
You are Larry King, a famous talk show host. 
You are interviewing a guest on your show.
The guest is a famous celebrity.
You ask the guest about their life, career, and upcoming projects.
The guest responds to your questions.
"""

@app.route('/stream', methods=['POST'])
def stream_response():
    input_data = request.json
    chat_history = ""
    for message in input_data['messages']:
        speaker = message['speaker']
        text = message['text']
        chat_history += f"<{speaker}> {text}</{speaker}>\n"
    
    chatml_prompt = f"<system>{PROMPT}</system>\n{chat_history}"
    inputs = tokenizer.encode(chatml_prompt, return_tensors='pt')

    # Generate response from model
    output_sequences = model.generate(inputs, max_length=1000)

    # Decode response
    # Stream the response
    def generate():
        all_text = ''
        for token in output_sequences[0]:
            out = tokenizer.decode(token, skip_special_tokens=True)
            all_text += out
            yield out
        print(all_text)
    
    return Response(generate(), mimetype="text/plain")

if __name__ == '__main__':
    app.run(debug=True)
