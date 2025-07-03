Output demo.mp4

Require just

Set FontSize 20
Set Width 1280
Set Height 720
Set FontFamily "JetBrainsMono Nerd Font"
Set Theme { "name": "SurrealDB", "black": "#15131D", "red": "#ef6487", "green": "#5eca89", "yellow": "#fdd877", "blue": "#65aef7", "magenta": "#FF00A0", "cyan": "#43c1be", "white": "#F9F9F9", "brightBlack": "#15131D", "brightRed": "#ef6487", "brightGreen": "#5eca89", "brightYellow": "#fdd877", "brightBlue": "#65aef7", "brightMagenta": "#aa7ff0", "brightCyan": "#43c1be", "brightWhite": "#F9F9F9", "background": "#29283b", "foreground": "#b3b0d6", "selection": "#3d3c58", "cursor": "#b3b0d6" }
Set MarginFill "#0E0C14"
Set BorderRadius 0
Set Margin 0
Set Padding 16
Set WindowBar Colorful
Set WindowBarSize 64
Set Framerate 20

Hide
Type "just docker-run" Enter
Type "tmux" Enter
Type "ollama serve" Enter
Ctrl+b
Type "c"
Type 'PS1="\033[38;2;255;0;160m> \033[m"' Enter
Type "clear" Enter

Show
Type "# First, let's start SurrealDB"
Sleep 500ms
Enter
Type "surreal start -u root -p root"
Sleep 500ms
Enter
Sleep 2s

Hide
Ctrl+b
Type "c"
Type 'PS1="\033[38;2;255;0;160m> \033[m"' Enter
Type "clear" Enter

Show
Type "# Now, let's init the project and install deps"
Sleep 500ms
Enter
Type "uv init --app" Enter
Type "uv add langchain_ollama surrealdb" Enter
Type "uv add git+https://github.com/surrealdb/langchain-surrealdb" Enter
Sleep 500ms
Type "clear" Enter
Type "# We are ready to code!"
Sleep 500ms
Enter
Type "vim main.py"
Sleep 500ms
Enter
Type "dGi"
Sleep 100ms
Type@10ms 'from langchain_core.documents import Document' Enter
Type@10ms 'from langchain_ollama import OllamaEmbeddings' Enter
Type@10ms 'from surrealdb import Surreal' Enter
Enter
Type@10ms 'from langchain_surrealdb.vectorstores import SurrealDBVectorStore' Enter
Enter
Type@10ms 'conn = Surreal("ws://localhost:8000/rpc")' Enter
Type@10ms 'conn.signin({"username": "root", "password": "root"})' Enter
Type@10ms 'conn.use("langchain", "example-basic")' Enter
Type@10ms 'vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)' Enter
Enter
Type@10ms '_url = "https://surrealdb.com"' Enter
Type@10ms 'd1 = Document(page_content="foo", metadata={"source": _url})' Enter
Type@10ms 'd2 = Document(page_content="SurrealDB", metadata={"source": _url})' Enter
Type@10ms 'd3 = Document(page_content="bar", metadata={"source": "https://example.com"})' Enter
Type@10ms 'd4 = Document(page_content="this is surreal", metadata={"source": _url})' Enter
Enter
Type@10ms 'vector_store.add_documents(documents=[d1, d2, d3, d4], ids=["1", "2", "3", "4"])' Enter
Enter
Type@10ms 'q = "surreal"' Enter
Type@10ms 'filter = {"source": _url}' Enter
Type@10ms `print(f'\nSearch: "{q}" filter {filter}\n')  # noqa: T201` Enter
Type@10ms 'results = vector_store.similarity_search_with_score(query=q, k=2, custom_filter=filter)' Enter
Type@10ms 'for doc, score in results:' Enter
Type@10ms 'print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201' Enter
Enter
Escape
Sleep 2s
Type ":wq" Enter
Sleep 100ms
Type "# Let's run it!"
Sleep 500ms
Enter
Type "uv run main.py"
Sleep 500ms
Enter
Sleep 5s
