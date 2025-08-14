def create_cache(self):
        """
        Creates a folder data_cache file with fraudDataset.csv file in it
        where further will be stored the dataset
        """
        project_root = Path(__file__).resolve().parents[1]
        cache_dir = project_root / 'data_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / 'fraudDataset.csv'


    def load_or_read_dataset(self, dataset):
        """
        If CSV exists and is not empty, read it.
        If CSV doesn't exist or is empty, load data and save it.
        """
        cache_file = self.create_cache()
        if cache_file.exists() and cache_file.stat().st_size > 0:
            self.df = pd.read_csv(cache_file)
            return self.df

        df = pd.read_csv(dataset)
        df.to_csv(cache_file, index=False)
        self.df = df
        return self.df