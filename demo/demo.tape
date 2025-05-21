Output demo/demo.webm

Require echo
Require just

Set FontSize 20
Set Width 1280
Set Height 720
Set FontFamily "JetBrainsMono Nerd Font"
Set Theme { "name": "SurrealDB", "black": "#15131D", "red": "#ef6487", "green": "#5eca89", "yellow": "#fdd877", "blue": "#65aef7", "magenta": "#FF00A0", "cyan": "#43c1be", "white": "#F9F9F9", "brightBlack": "#15131D", "brightRed": "#ef6487", "brightGreen": "#5eca89", "brightYellow": "#fdd877", "brightBlue": "#65aef7", "brightMagenta": "#aa7ff0", "brightCyan": "#43c1be", "brightWhite": "#F9F9F9", "background": "#29283b", "foreground": "#b3b0d6", "selection": "#3d3c58", "cursor": "#b3b0d6" }
Set MarginFill "#0E0C14"
Set BorderRadius 32
Set Margin 32
Set Padding 32
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
Type@10ms "from langchain_core.documents import Document" Enter
Type@10ms "from langchain_surrealdb.vectorstores import SurrealDBVectorStore" Enter
Type@10ms "from langchain_ollama import OllamaEmbeddings" Enter
Type@10ms "from surrealdb import Surreal" Enter
Enter
Type@10ms 'conn = Surreal("ws://localhost:8000/rpc")' Enter
Type@10ms 'conn.signin({"username": "root", "password": "root"})' Enter
Type@10ms 'conn.use("langchain", "demo")' Enter
Type@10ms 'vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)' Enter
Enter
Type@10ms 'print("Adding docs...")' Enter
Type@10ms 'doc_1 = Document(page_content="foo")' Enter
Type@10ms 'doc_2 = Document(page_content="SurrealDB")' Enter
Type@10ms 'vector_store.add_documents(documents=[doc_1, doc_2], ids=["1", "2"])' Enter
Enter
Type@10ms 'print("Searching \"surreal\"...")' Enter
Type@10ms 'results = vector_store.similarity_search_with_score(' Enter
Type@10ms 'query="surreal", k=1' Enter
Type@10ms ")" Enter
Type@10ms "for doc, score in results:" Enter
Type@10ms 'print(f"[similarity={score:3f}] {doc.page_content}")' Enter
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
