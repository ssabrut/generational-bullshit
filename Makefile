BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
REMOTE := origin

.PHONY: help pull push sync status log branch checkout merge pr clean

help:
	@echo "Usage: make <target> [msg='commit message'] [branch=<name>] [into=<name>]"
	@echo ""
	@echo "  pull              Pull latest from current branch"
	@echo "  push msg='...'    Stage all, commit, and push to current branch"
	@echo "  sync              Pull then push (rebase-based sync)"
	@echo "  status            Git status"
	@echo "  log               Last 10 commits (graph)"
	@echo "  branch            List all branches"
	@echo "  checkout b=<name> Switch to branch (creates if new)"
	@echo "  merge into=<name> Merge current branch into target"
	@echo "  pr                Open a pull request via gh CLI"
	@echo "  clean             Remove merged local branches"

pull:
	git pull $(REMOTE) $(BRANCH) --rebase

push:
	@if [ -z "$(msg)" ]; then echo "Error: provide msg='your commit message'"; exit 1; fi
	git add -A
	git commit -m "$(msg)"
	git push $(REMOTE) $(BRANCH)

sync: pull push

status:
	git status

log:
	git log --oneline --graph --decorate -10

branch:
	git branch -a

checkout:
	@if [ -z "$(b)" ]; then echo "Error: provide b=<branch-name>"; exit 1; fi
	git checkout -B $(b)

merge:
	@if [ -z "$(into)" ]; then echo "Error: provide into=<target-branch>"; exit 1; fi
	git checkout $(into)
	git merge $(BRANCH) --no-ff
	git checkout $(BRANCH)

pr:
	gh pr create --fill

clean:
	git branch --merged | grep -v "\*\|main\|master\|develop" | xargs -r git branch -d
