#!/usr/local/bin/python3
import argparse
import os
import subprocess

lesson_names_lst = [
        '1-Intro',
        '2-Corpus-Linguistics',
        '3-Clustering-and-Topic-Modeling',
        '4-Word-Embedding',
        '5-Reliability',
        '6-Classification',
        '7-Information-Extraction',
        '8-Semantic-Networks',
        '9-Beyond-Text',
]

lesson_names = {str(i + 1) : n for i, n in enumerate(lesson_names_lst)}

special_names = {
    'all' : lesson_names_lst,
}

def getArgs():
    parser = argparse.ArgumentParser(description='Helper for managing conversion between notebooks and markdown')
    parser.add_argument("--verbose", "-v", action = 'store_true', default = False, help = "Verbose mode, every step is printed")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-notebook", "-n",  default = False, action = 'store_true', help = "Convert named lessons to notebooks")
    group.add_argument("--to-markdown", "-m",  default = False, action = 'store_true', help = "Convert named lessons to markdown")
    parser.add_argument("--run", "-r",  default = False, action = 'store_true', help = "Run the notebooks once generated")
    parser.add_argument("names", default = [], nargs = '+')
    return parser.parse_args()

def name_converter(name):
    if name in lesson_names_lst:
        return name
    elif name.split('.')[0] in lesson_names_lst:
        return name.split('.')[0]
    elif name in lesson_names:
        return lesson_names[name]
    else:
        raise argparse.ArgumentTypeError('Invalid target name: {}'.format(name))

def genNotebook(name, run = False, verbose = False):
    os.makedirs(name, exist_ok = True)
    inputFile = f'{name}/{name}.md'
    outputFile = f'{name}/{name}.ipynb'
    if verbose:
        print(f"Creating {outputFile}")
    subprocess.run(['notedown', '-o', outputFile, inputFile])
    if run:
        if os.path.isfile(inputFile):
            if verbose:
                print(f"Running {outputFile}")
            subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--output={}'.format(outputFile), outputFile])
        else:
            print(f"Missing {inputFile}, skipping {outputFile}")

def genMarkdown(name, verbose = False):
    inputFile = f'{name}/{name}.ipynb'
    outputFile = f'{name}/{name}.md'
    if os.path.isfile(inputFile):
        if verbose:
            print(f"Creating {outputFile}")
        subprocess.run(['notedown', '--strip', '-o', outputFile, inputFile])
    else:
        print(f"Missing {inputFile}, skipping {outputFile}")

def main():
    args = getArgs()
    if len(args.names) == 1 and args.names[0] in special_names:
        names = special_names[args.names[0]]
    else:
        try:
            names = [name_converter(n) for n in args.names]
        except argparse.ArgumentTypeError as e:
            print(e)
            print("Given: {}".format(' '.join(args.names)))
            return
    if args.verbose:
        print("Given: {}".format(' '.join(names)))
    for name in names:
        if args.to_markdown:
            genMarkdown(name, verbose = args.verbose)
        elif args.to_notebook:
            genNotebook(name, run = args.run, verbose = args.verbose)
    print(args)

if __name__ == '__main__':
    main()
