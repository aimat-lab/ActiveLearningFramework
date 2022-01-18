class CandidateUpdater:
    """
    Responsible for updating the candidate set (providing candidates for the query selection)
    """

    def update_candidate_set(self):
        """
        Scenario dependent: get instance(s) from candidate_source (Pool/Stream/Generator), add predictions from current PL and insert instance into candidate set

        :raise NoMoreCandidatesException: if no new element can be fetched from candidate source
        """
        raise NotImplementedError
