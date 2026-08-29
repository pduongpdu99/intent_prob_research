from engine.intent import Brain, join

brain = Brain()
brain.learn(join("engine", "intent", "data", "tu_van_phan_mem.aiml"))

while True:
    ask = input("You: ")
    print("Bot:",brain.respond(ask))
