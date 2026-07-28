import ollama

#print(ollama.list())
#print(ollama.show("llama3.1")["modelfile"])

ollama.create(model="Imperator", 
              from_ = "llama3.1:latest",
              system = "You are a very smart assistant who knows everything about the Roman Empire. You are very succinct, and informative.",
              parameters = {"temperature":5})


def OllamaTalk(request):
    talk = ollama.chat(
        model = "Imperator",
        messages =[
            {"role":"user","content":f"{request}"}
        ]
    )
    return talk["message"]["content"]

print(OllamaTalk("How big was the Roman Empire?"))

ollama.delete("Imperator")