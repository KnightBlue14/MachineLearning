import ollama

model_list = ollama.list()
#print(model_list)

# print(ollama.show("llama3.1")["modelinfo"])
# print(ollama.show("llama3.1")["modelinfo"]["general.license"],
#      ollama.show("llama3.1")["details"]["parameter_size"])

prompt = 'tell me a fun fact about the Roman Empire'
model = "llama3.1"

def OllamaTalk(request,model):
    talk = ollama.chat(
        model = f"{model}",
        messages =[
            {"role":"user","content":f"{request}"}
        ]
    )
    return talk["message"]["content"]

#print(OllamaTalk(prompt,model))

def OllamaTalkstream(request,model):
    talk = ollama.chat(
        model = f"{model}",
        messages =[
            {"role":"user","content":f"{request}"}
        ],
        stream = True
    )
    for chunk in talk:
        print(chunk["message"]["content"], end="", flush=True)

#print(OllamaTalkstream(prompt))
