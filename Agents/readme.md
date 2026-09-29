# Agentic AI

So far, mainly in the Ollama library, I've been showing how to leverage local AI to power local applications, such as RAG pipelines. However, not only have the examples provided been limited to singlular uses, they have been mainly using larger models to enable more complex functions, which is relatively inefficient for smaller workloads. In this library, I'll be going over how to build tools to empower models to perform similar tasks much more efficiently.

## What is an agent?

To quickly explain, agentic AI recently become the standard for deploying LLMs. Natively, they can be equipped with certain functions, such as image analysis, but their ability to interact with their environment beyond that is limited. Tools such as MCP servers and harnesses were then developed in order to provide LLMs with 'tools' - deterministic functions that can then be leveraged to enhance the capabilities of LLMs, allowing them to now perform more advanced tasks, including web searching, file analysis, and even generating new files, now readily seen from larger LLM providers such as Anthropic. When used properly, this allows for workloads to take advantag of the liberty of language-based instruction, but processing those instructions through a specifically configured function, preventing the LLM from causing unwanted damage. This process has been somewhat automated by now, with tools such as n8n and harnesses bundled with the tools, but it is useful to be able to define your own functions.

## Basic agent

To begin, in agent_tools.ipynb, I have included a prompt to a small model, qwen3:1.7b, to demonstrate a native function, in this case describing the size of the Roman Empire. The model's training data included this information, so it is easily able to answer the question in detail.

I also have a file, fox.txt, which includes the phrase 'The quick brown fox jumps over the lazy dog' (a pangram, meaning it uses every letter of the alphabet at least once). Prompting the model to summarise the file, it responds that it cannot read the file, as it was not included in the model's context. Therefore, we need to find a way to add the file to the context, allowing the model to read it.

Before, I used an RAG pipeline to do this, but is very computationaly expensive, and needs other packages, such as chroma, adding to the bloat for a very simple task. Thankfully, we can also use a much more efficient method.

To begin, we define a function, as we would normally. Importantly, we are returning the now opened file, rather than making the content readable to the user, as we need it loaded into memory for the model to be able to use it. That done, we make this function into a tool. 

How exactly this is done will vary between your provider. In my case, I don't actually need a tool schema, since my setup is entirely local, but if you use ChatGPT or Claude Code, you may need to provide one to pass through their API. I've included it here as a reference to how it should look, including what it is, what function is being called, how it carries out the function, and what is required for it to function. Running it locally, meanwhile, I just need to declare it as a tool later.

Next, we set up the 'messages' list. This will set up the model's conversation memory, allowing us to refer to the results of previous prompts. I don't use it much here, but it is useful to have for longer sessions.

Then, we set up the response function. Here, we input our message, then submit which tool to use (our read_file function), and the LLM responds with the content of the file. Below, we can see in the model's thinking process that it is recognising the tool, what it does, and what components are needed to complete the task. We can also confirm that the file is loaded into memory, and the LLM can perform tasks with it's contents, by having it count the number of words in the sentence. We have provided no function for this, but it correctly counts them, leveraging the deterministic function of reading the file with the language comprehension of the LLM. All of the usual warnings do still apply, but this offers some railguards to prevent issues.