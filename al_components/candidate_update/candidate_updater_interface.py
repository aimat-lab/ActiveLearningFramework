class CandidateUpdater:
    """
    Responsible for updating the candidate set (providing candidates for the query selection)
    """

    def update_candidate_set(self):
        """
        Scenario dependent: get instance(s) from candidate_source (Pool/Stream/Generator), adds predictions from current PL and inserts instance into candidate set

        :raise EndTrainingException: if no new element can be fetched from candidate source
        """
        # TODO: raising of EndTraining appropriate??
        raise NotImplementedError
