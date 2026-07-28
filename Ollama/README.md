
# Ollama

This directory covers the use of Ollama, an open source tool for downloading and running AI models locally, as well as modifying them to your use case. In particular, this will cover using python to interact with Ollama via the inbuilt API.


## Description

Ollama is a free tool to download and run AI models on local hardware, not requiring a conncetion to a server, as is the case with paid services such as ChatGPT. In conjunction with another tool called OpenWebUI, this allows you to run a local LLM with a user friendly web interface, from a selection of user uploaded models. You can also select from a variety of sizes of model, meaning it can also be run on less powerful hardware, though this will affect the overall quality of the output. I'll briefly go over how to install it, then cover using python to interact with it.
## Installing Ollama

Ollama can be downloaded from the website

https://ollama.com/download

On Windows and MacOS, you will download an installer, and on Linux you will run a script to download and install. In this case, I will be using Linux commands.

That done, you can download models from the terminal. To find models, you can go to the website to find the model that suits your needs.

https://ollama.com/search?q=llama

A property to keep in mind is the parameter size. This refers to the amount of information used to train the model, and in turn the cost of running the model, in billions of parameters. Smaller models, such as llama3.2, can be run on weaker hardware, but have less information, limiting their effectiveness.
Here, I will use llama3.1, as the 8b parameter size is a good compromise between demands and ability.

To download, simply run from the terminal
```
ollama pull llama3.1:latest
```
Which, in this instance, is the 8b parameter model.

That done, we can then type
```
ollama run llama3.1
```
to interact with our model.

From here, we can also install OpenWebUI to add a web interface, but I will not be covering that here.
## Ollama in Python

### API

Ollama functions using an internal API, running on localhost:11434. You can go to this address in a web browser to make sure Ollama is working.

This also means that we can interact with it via this API, using the requests module in Python, as shown in restapi.py.

Here, we can feed a dictionary to Ollama, telling it which model to use and a prompt - in this case, asking it to tell us a fact about the Roman Empire. Of note is that in the 'response' variable, I have set the 'stream' property to True. This will set the reponse to return in realtime, rather than waiting for it to complete, much like how ChatGPT does. If that is to your preference, make sure stream is set to True.

From here, we then need to decode and format the response, which will print in the form of an answer to our question.
### Ollama library

Moving on to llamalibrary.py, we can begin to interact with our Ollama model in more interesting ways. Firstly, we can directly grab information, without using the requests module. For instance, we can use ollama.list() to print information about all of our models, though still not in a very user friendly format.

If we just want the name and size of our model, we can use ollama.show() to provide information about a specific model, then select specific information, resulting in a printout like this -
```
llama3.1 8.0B
```

Much like before, we can also send prompts to our model, and adjust how it will respond. After assigning our prompt and model to variables (trust me, this will save a lot of time), we can create a function that will push the prompt to that model.

In the first block, the response will first buffer before being printed out, while the second will stream. Again, which you use will depend on your preference.

### Building a model

Another feature of the Ollama library is creating new models, using 
```
ollama.create()
```
This will allow us to take a pre-existing model, and customise it using the system instruction, basically a prompt, like those we used earlier, but for every time the model is used. 
In modelsetup.py, I have set up a model called Imperator, an LLM focused on the Roman Empire. There are many properties that we can adjust to change the model's behaviour, but to start with we'll just use a few.

Do note that while I have limited the focus of the LLM, which we'll see an example of below, the model is still the same size as the original llama3.1. All of the original information is still there, and I havn't added or removed anything. Keep this in mind if storage space is limited, as you will be doubling the space taken by models whenever you create one using this method.

Within the create command, we specify the name, Imperator, and the model, llama3.1. Then, we enter our system prompt, telling it to focus on the Roman Empire, and how it should answer any prompts. We can also give more specific instructions, such as answers including a phrase or being in a certain format. Finally, we will input a parameter, in this case the temperature. 

It is important to remember that an LLM is not actually 'thinking' about the answer - it is selecting words based on probability, based on the information used to train it. If I say 
```
I have a fluffy ___
```
 the LLM will respond 'cat' because, in the training data, the word 'fluffy' was followed by 'cat' most of the time.

Back to temperature - a model's temperature is used to describe how 'verbose' it's responses can be. As the temperature is increased, the range of words available will be more evenly distributed, meaning more words become more likely to follow the prompt. So, initially, every time I input the earlier prompt, with a low temperature, the LLM will always respond with 'cat'. However, as I increase the temperature, there will be more variety, adding new words, such as 'rabbit' or 'dog'.

Examples of this can be found in responses.txt - given the prompt 
```
How big was the Roman Empire?
```
Imperator gave a number of different responses, with only the temperature being changed.
At a low temperature, 0.01 and 0.1, the answers are very similar, in both structure and specificity. At 0.5 and 1, it is still similar, but is less specific, not offering a full list of territories, and rounding to the nearest million kilometres at the higher temperature. Finally, at 5, it offers a very elaborate response, providing information not directly relevant to the answer, such as which Roman regions coorespond to which modern region and how they were governed.

Adjusting the temperature and the system prompt will allow you to modify a given LLM to respond in line with your particular needs, allowing you to make the response more clinical and direct, or more in-depth, and with extra information. You can even have it not respond to certain prompts, such as in the bottom of responses.txt, where Imperator could not tell me who Michael Jackson was. Such a limit could be imposed indirectly, by telling it to only focus on a particular subject, or by specifying that it will not answer questions about a specific topic. 

### Importing documents

That's all well and good, but all we've done is modify the behaviour of a pre-existing model. We can still only ask it to provide information it was trained on, meaning that if we ask it a question about something from after it was trained, or was too niche for it to be selected, the LLM will not be able to provide any information. We need to append the information before it can be used

In this case, I will have the AI 'read' a report from the UK government from after the Covid-19 pandemic, addressing the economic rebuilding strategy with regard to net zero and sustainable, ecologicaly minded practices. To do this, we will need to not only upload a copy of the report, but also build a database that a model can use to answer questions.

After installing all the libraries in requirements.txt and setting up imports in readdocs.py, we set the path to our file, and which model we wish to use. Again, I will be using llama3.1.

We will also need to download a new model - an embedding model, the purpose of which is to transform the report into a medium that can be read by our LLM. In this case, I will be using nomic-embed-text, which you can install the same way you installed the earlier model.

That done, we import our file via the loader, and split it into chunks (Line 18-31). Each entry in a vector database can only be a given size, so we must make each part of the report smaller. The chunk size controls how much text is allocated to each chunk, and overlap helps to preserve context for chunks, which is vital for the LLM interpreting the document. In this case, with the settings listed, a 90 page document is broken into -
```
Number of chunks: 298
```

From there, we need to build a vector database in which to store our chunks. This is a simple block, as we just need to allocate the collection of chunks, and the model to be used in embedding. (Lines 33-39)

Now we create an interface for reading the file - essentially, a temporary model that is not kept in storage. Like before, we create a system prompt to define the behaviour. Then, create a retriever to allow the model to gather information, as well as a template to restrict the output to only read from that information. Finally, we bring it together in a chain, defining a workflow of prompting, collecting and responding.

Now that we've set up our interface, it's time to ask some questions. Here, I've set it to save the questions and answers to readreport.txt. 

Of particular note is that, compared to before, the hardware requirements are much more noticable. You may have noticed that I have included two lines of code to note the time, as a way to benchmark the model. Also note, I am on my laptop using an IGPU, so performance was always going to be limited. That said, to build the database and parse the report, it took -

```
951.3307294845581 seconds
```
or over 15 minutes. This is where using a dedicated GPU, particularly one that uses CUDA, will be a benefit.

And with that, we have successfully used an LLM to 'read' and summarise a document, also known as Retrieval Augmented Generation (RAG). Using this method, we can provide the LLM with documents and guides, allowing it to (relatively) quickly scan through many pages across multiple files to answer questions.

A word of caution, however - even after years of development, and even in enterprise scenarios, such models have been known to misinterpret and hallucinate both prompts, and the results. What the model returns should not be taken for granted, and any vital information should always be verified before being acted on.


### Halloween Special

As it is October, I decided to take the opportunity to do something a bit more on the fun side - use the RAG method to ask questions about Mary Shelley's Frankenstein. (Spoilers, it's not very good. Read on to find out why)

First, we need a copy of the book. If you don't already have a digital copy, check out Project Gutenberg
```
(https://www.gutenberg.org)
```
This is a collection of free ebooks, mostly public domain works, built to encourage the preservation and distribution of these works. 

With it downloaded into our active directory, we can then build another RAG pipeline, adjusting the system prompt, and asking it our questions. 

In readbook.txt, we can see that the results are initially promising, as the model is able to successfully summarise the events of the book, as well as the themes. However, upon asking it the name of Victor's cousin (Elizabeth Lavenza), it instead responds with Henry Clerval, his childhood friend. Similarly, in earlier test runs, it would respond with other characters, rather than Elizabeth, who, in Chapter 6, writes Victor a letter, opening with the words 'My dearest cousin'.

All in all, this was a partial success, as the model did summarise the events and themes of the book, though it had trouble identifying some key pieces of information. Also, as mentioned, this is still based on what we have provided, so it will not be able to provide, for example, the history of the author, or contemporary opinions when it  was written. If you were so inclined, you would need to also provide these in the database.

--UPDATE - Running the script on my desktop, with an RTX 2060 6GB GPU, the results are much better - not only is it  faster, at under 4 minutes, the model correctly identifies Elizabeth as Victor's cousin. That said, the novel is already embedded into llama3.1, so it is possible that my system prompt did not completely prevent it from using information outside the provided book. Either way, goes to show that running models with a GPU is prefereable to without one.

### Persisting RAG stream

While our RAG pipeline works (depending on our use case), it's still rather awkward to use, running the file manually each time we want to ask a question, and having to build a new database each time. To wrap up, we'll build another pipeline with a persisting database, and even one that iterates over multiple files.

For this, we'll be using a new library called streamlit. It takes very litle setup, and can easily incorporate our model's output. Beyond that, setup is much the same for single file ingestion - looking at rag_stream.py, we simply clean up the code a bit by incorporating functions, and setting the file to run on repeat with the 'main' loop at the end.

For multi-file setup, this, again, is fairly simple. We just need to keep all of our reports in a folder, and point the DirectoryLoader at that folder, which then runs the PDFLoader on every file.

Finally, in order to run the file, we can't run it via python. Instead we need to use the command
```
streamlit run rag_stream.py or streamlit run rag_stream_multi.py
```

This will then open a window in your browser, where you can then ask the model your questions.

As for results, the quality is overll mixed. While it does save some time not needing to rebuild the database, querys can still take over 15 minutes. This is a hardware limitation on my end, but it is something to be aware of if you're in a similar position. 

--UPDATE - after importing the script onto my main desktop, with an RTX 2060 6GB GPU, a similar query for the multi file stream can build the database and be answered in less than 5 minutes, showing that hardware does indeed make a major difference in speed. Also, I have updated the imports, as my laptop was using an earlier version of langchain, and the file paths have been updated since then. Apologies for any confusion this may have caused.

The results themselves are also a mixed bag. Single file prompts work well, provided the file is one that provides full context, such as a government report or scientific paper. The multi-file prompts, however, leave something to be desired - asking the model specific questions about the contents of the reports work just fine. For instance, it was able to correctly describe the process of taking gas meter readings, described in one of the papers, even combining multiple small steps into a larger one. However, when asked to summarise the ESOS scheme, it did descrive multiple key points correctly, but fabricated that business need to consume a certain amount of power each year, which is not mentioned in that paper. Finally, when asked to list the documents in the directory, it would identify sections of a report as multiple reports, as additional metadata would be needed to identify each report. Rather than breaking down each report into it's own database, we have instead broken them into the same pile of chunks, which then were built into one database, so there is nothing to distinguish each file.

--UPDATE - Also, after some additional prompting, I've found that despite me telling the model to only use the context (that is, whichever files I've loaded into the vector database), the model can still use outside information, if it determines that the context is not enough to answer a question. You can specify in the RAG template further, as I have done, but that will affect the quality of your answers, if it cannot properly read the documents provided. Finally, to add to the paragraph above, using a more powerful model can improve performance substantially - running readdocs with a larger gemma model and the gemma embedding model greatly improved the results, being able to correctly identify subsections of a document, and sometimes the correct page, though not always

All in all, it is worth remembering that LLMs are not 'thinking' models (even if designated as such, like with deepseek-r1 - I know this because the results from that, if anything, were worse). They are algorithms picking the next most likely word. If you choose your use case wisely, they can work well, but, as 95% of ventures have recently realised, they are not miracle makers. You must be clinical with your querys, and precise with the information you want, as well as making sure that the model is provided all of the information it needs.

As long as you keep that in mind, you can begin to use such tools more responsibly. And thanks to open source ventures such as Ollama, you can do so on your own terms, on your own hardware, without having to rely on online services such as ChatGPT.