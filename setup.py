import setuptools

setuptools.setup(
        name = 'fcgtools',
        version = '0.1.0',
        packages = setuptools.find_packages(),
        install_requires=[
                'tqdm',
                'pyyaml',
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

