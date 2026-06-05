class Report:
    def __init__(self, **kwargs) -> None:
        self.names = self.__dict__

        self.diagnose = kwargs.get('diagnose', dict(
            T=None,
            p_clean=None,
            p_org=None,
            class_distribution=None,
            group_distribution=None,
        ))
        self.detection = kwargs.get('detection', dict(
            score_error=None,
            coexistence=None,
            rare_example=None,
        ))
        self.curation = kwargs.get('curation', dict(
            score_curation=None,
            sampling_strategy=None,
            feature_curation=None,
        ))
        self.audition = kwargs.get('audition', dict(
            model_perf=None,
            fairness=None,
            stress_test=None,
        ))

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict) and key in self.names and isinstance(self.names[key], dict):
                self._update_dict(self.names[key], value)
            else:
                self.names[key] = value

    def _update_dict(self, original, updates):
        for k, v in updates.items():
            original[k] = v
