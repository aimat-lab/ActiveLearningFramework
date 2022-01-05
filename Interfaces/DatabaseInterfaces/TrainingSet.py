class TrainingSet:

    def append_labelled_instance(self, x, y):
        raise NotImplementedError

    def pop_all_labelled_instances(self):
        raise NotImplementedError
