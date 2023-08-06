import glob

from Preprocessing import split_content

if __name__ == "__main__":
    print("Logs: Retrieving files recursively")
    dirs_of_tf_files = glob.glob("**/*.tf", recursive=True)
    split_content( dirs_of_tf_files)