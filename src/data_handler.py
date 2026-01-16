

class DataHandler:
    def __init__(self, df_map):
        self.df_map = df_map

        self.sanity_check()

        self.train_df_map = None
        self.test_df_map = None
    
    def get_keys(self):
        return list(self.df_map.keys())
    
    def sanity_check(self):
        