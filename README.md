# dealmonitor_ml
The ML Model to power the dealmonitor as well as the training and stuff.

## Run a training session
Because training takes time and ssh tends to stop processes when the connection times out, it is wisely recommended to use a tmux
training session.

```bash
tmux new -s training_session
# Starte Trainings script, e. g.:
python train_nn_model.py
```

detach training session using 'ctrl + b' and afterwards 'd'

reattach session using:
```bash
tmux attach -t training_session
```

## Including the Dealmonitor Repository as a Submodule
This project uses Git submodules to include the Dealmonitor repository within the dealmonitor_ml project. Git submodules allow you to keep a separate repository as a subdirectory in your project while maintaining independent versioning.

### Adding the Submodule
1. Open a terminal and navigate to the root directory of your dealmonitor_ml repository.

2. Add the Dealmonitor repository as a submodule (replace the URL with the actual URL of your dealmonitor repo):
```bash
git submodule add git@github.com:rwkaspar/dealmonitor.git dealmonitor
```
This will create a folder named dealmonitor containing the Dealmonitor repository.

3. Commit the changes:
```bash
git add .gitmodules dealmonitor
git commit -m "Add Dealmonitor repository as a submodule"
```

### Cloning the Repository with Submodules
When cloning this repository for the first time, you need to initialize and fetch the submodules separately:

```bash
git clone https://github.com/yourusername/dealmonitor_ml.git
cd dealmonitor_ml
git submodule update --init --recursive
```
This ensures the Dealmonitor submodule contents are also downloaded.

### Updating the Submodule
To pull new changes from the Dealmonitor submodule’s remote repository:

```bash
cd dealmonitor
git pull origin main
cd ..
git add dealmonitor
git commit -m "Update Dealmonitor submodule"
```
Using Git submodules keeps the Dealmonitor repository cleanly separate but integrated into the ML project, allowing independent version control and easier maintenance.

For more information on Git submodules, see the official Git documentation: https://git-scm.com/book/en/v2/Git-Tools-Submodules