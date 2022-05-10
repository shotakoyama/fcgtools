import setuptools

setuptools.setup(
        name = 'fcgtools',
        version = '1.0.0',
        author = 'Shota Koyama',
        packages = setuptools.find_packages(),
        install_requires = [
            'lemminflect>=0.2.2',
            'numpy>=1.22.3',
            'sacrebleu>=2.0.0',
            'sacremoses>=0.0.49',
            'sentencepiece>=0.1.96',
            'tabulate>=0.8.9',
            'tokenizers==0.12.1',
            'torch',
            'tqdm==4.64.0',
            'transformers==4.18.0',
        ],
        entry_points = {
            'console_scripts':[
                'fcg-prepare = fcgtools.cli.prepare:main',
                'fcg-preproc = fcgtools.cli.preproc:main',
                'fcg-apply-m2 = fcgtools.cli.apply_m2:main',
                'fcg-train = fcgtools.cli.train:main',
                'fcg-generate = fcgtools.cli.generate:main',
                'fcg-filter = fcgtools.cli.filter:main',
                'fcg-score = fcgtools.cli.score:main',
                ]},)

