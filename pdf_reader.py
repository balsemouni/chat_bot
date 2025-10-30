import ollama
history=[]
model="phi3:mini"
while True:
    prompt=input("You: ").strip()
    if prompt.lower() in ("exit","quit"):
        print("have a nice day!")
        break

    message={
        "role":"user",
        "content":prompt
    }
    history.append(message)
    print("Bot:",end="",flush=True )
    bot_message_content=""
    response=ollama.chat(model=model,messages=history,stream=True)
    for chunk in response :
        bot_message_content+=chunk.message.content
        print(chunk.message.content,end="",flush=True)
        
    print(f"Bot:{bot_message_content}")
    bot_message={
        "role":"assistant",
        "content":bot_message_content
    }
    history.append(bot_message)
