from render_sdk import Workflows

app = Workflows()

@app.task()
def start_flask(a: int) -> int:
    return a * a

if __name__ == "__main__":
    app.start()
