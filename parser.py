import argparse

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-public', '--public', type=int, default=0, help='share chat bot interface')
    parser.add_argument('-domain', '--domain', type=int, default=0, help='0: music, 1: disease')
    parser.add_argument('-model_ollama', '--model_ollama', type=int, default=4, help='select model for ollama')

    args = parser.parse_args()

    return args 