import ollama

MODEL = "llama3.2:3b"
TURNS = 3
TOPIC = "Start a casual conversation about any random topic you like."

AGENTS = {
    "Alice": "You are Alice, a curious and enthusiastic person. Keep replies to 1-2 short sentences.",
    "Bob": "You are Bob, a dry-witted skeptic. Keep replies to 1-2 short sentences.",
}


def reply(name, transcript):
    """Ask `name` to respond, given the shared transcript of (speaker, text) pairs."""
    # Each agent sees its own lines as "assistant" and the other's as "user".
    messages = [{"role": "system", "content": AGENTS[name]}]
    if not transcript:
        messages.append({"role": "user", "content": TOPIC})
    for speaker, text in transcript:
        messages.append({"role": "assistant" if speaker == name else "user", "content": text})
    return ollama.chat(model=MODEL, messages=messages)["message"]["content"].strip()


def main():
    ollama.pull(MODEL)  # no-op if already present
    transcript = []
    names = list(AGENTS)
    for turn in range(TURNS):
        for name in names:
            text = reply(name, transcript)
            transcript.append((name, text))
            print(f"[{turn + 1}] {name}: {text}\n")


if __name__ == "__main__":
    main()
