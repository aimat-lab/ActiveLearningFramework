class CandidateSet:

    def add_instance(self, x, y_prediction, accuracy):
        raise NotImplementedError

    def retrieve_all_instances(self):
        raise NotImplementedError

    def remove_instance(self, x):
        raise NotImplementedError

    def update_instance(self, x, new_y_prediction, new_certainty):
        raise NotImplementedError
