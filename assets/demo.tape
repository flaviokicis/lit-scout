# Re-record: mkdir demo-research (add a CLAUDE.md telling the model to answer without tools), trust it once
# in Claude Code, then run `vhs demo.tape` from the folder that holds demo-research.
Output demo.gif
Set Shell zsh
Set FontSize 16
Set Width 1300
Set Height 950
Set Framerate 10
Set Theme "Dracula"
Set TypingSpeed 20ms

# Startup and a filler answer stay hidden so no header or account detail is in frame
Hide
Type "cd demo-research && PS1='$ ' && unset CLAUDE_CODE_CHILD_SESSION && clear && claude --model sonnet" Enter
Sleep 9s
Type "Reply with the numbers 1 to 120, one per line, no tools, nothing else." Enter
Sleep 25s

# 1. Debugging prompt: no offer
Show
Type "the loss spikes at step 4k every run, can you check the data loader shuffling? answer in one line"
Sleep 600ms
Enter
Hide
Sleep 9s
Show
Sleep 2500ms

# 2. Optimizer idea: offer, press 2
Type "would it help to move the 1B model's hidden matrices to Muon and keep embeddings and norms on AdamW?"
Sleep 600ms
Enter
Hide
Sleep 6s
Show
Sleep 3s
Type "2"
Sleep 1s

# 3. RLHF idea: offer, press 0
Type "what if our RLHF run also penalized the spread among 4 reward models, on top of the KL term?"
Sleep 600ms
Enter
Hide
Sleep 6s
Show
Sleep 3s
Type "0"
Sleep 1s

# 4. Mech interp idea: offer, press 1, search, report, add findings to chat
Type "what if we fit a shared crosscoder on a few early checkpoints to find when induction heads form?"
Sleep 600ms
Enter
Hide
Sleep 6s
Show
Sleep 3s
Type "1"
Sleep 4s
Hide
Sleep 40s
Show
Sleep 7s
Type "1"
Sleep 2s
Type "ok, given those papers, what do I do first?"
Sleep 600ms
Enter
Hide
Sleep 10s
Show
Sleep 5s
