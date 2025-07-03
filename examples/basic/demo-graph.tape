Output demo-graph.mp4

Require just

Set FontSize 40
Set Width 2560
Set Height 1440
Set FontFamily "JetBrainsMono Nerd Font"
Set Theme { "name": "SurrealDB", "black": "#15131D", "red": "#ef6487", "green": "#5eca89", "yellow": "#fdd877", "blue": "#65aef7", "magenta": "#FF00A0", "cyan": "#43c1be", "white": "#F9F9F9", "brightBlack": "#15131D", "brightRed": "#ef6487", "brightGreen": "#5eca89", "brightYellow": "#fdd877", "brightBlue": "#65aef7", "brightMagenta": "#aa7ff0", "brightCyan": "#43c1be", "brightWhite": "#F9F9F9", "background": "#242133", "foreground": "#b3b0d6", "selection": "#3d3c58", "cursor": "#b3b0d6" }
Set MarginFill "#9600FF"
Set BorderRadius 0
Set Margin 0
Set Padding 16
# Set WindowBar Colorful
Set WindowBarSize 64
Set Framerate 20
Set PlaybackSpeed 1.0

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
Type "uv add langchain_ollama surrealdb langchain_community langchain_surrealdb" Enter
Sleep 500ms
Type "clear" Enter
Type "vim graph.py" Enter
Sleep 500ms
Type "/# -- " Enter
Sleep 2s
Type "n"
Sleep 2s
Type "n"
Sleep 2s
Type ":wq" Enter
Type "# Let's run it!"
Sleep 500ms
Enter
Type "uv run graph.py"
Sleep 500ms
Enter
Sleep 300s
