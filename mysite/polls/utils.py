from github import Github

def push_file(file_name):
    g = Github("991fc3ddb0607ff121af0b5197cfab3f1ec2d0c8")
    repo = None


    for repo in g.get_user().get_repos():
        if repo.name == "dataset_urdu_chars":
            break
    if repo is None:
        return

    with open(file_name, 'r') as file:
        content = file.read()

    # Upload to github
    git_prefix = 'data/'
    git_file = git_prefix + file_name
    repo.create_file(git_file, f"committing {file_name}", content, branch="main")
    print(git_file + ' CREATED')