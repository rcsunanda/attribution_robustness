# Class to calculate group errors for a dataset batch by batch
# Currently only supports 4 groups (binary label and binary attribute)
# group_key = label * 2 + attribute
# group_key range: 0 to 3 #### (0, 0) --> 0, (0, 1) --> 1, (1, 0) --> 2, (1, 1) --> 3

class GroupErrorCalculator:
    def __init__(self, device_obj):
        self.device_obj = device_obj
        # Group key: combine the binary label and binary attribute as below to get a unique key for each group
        # group_key range: 0 to 3 #### (0, 0) --> 0, (0, 1) --> 1, (1, 0) --> 2, (1, 1) --> 3
        self.group_keys = [0, 1, 2, 3]

        self.group_num_samples = {0: 0, 1: 0, 2: 0, 3: 0}
        self.group_num_incorrect_preds = {0: 0, 1: 0, 2: 0, 3: 0}

        self.is_error_computed = False  # This class only supports computing group errors once for a dataset

    def add_batch(self, metadata, labels, is_incorrect_preds):
        assert self.is_error_computed is False, 'Group errors have already been computed for this dataset'

        metadata = metadata[0]  # Get the metadata element from the above iterator's tuple (batch of metadata)
        metadata = metadata.to(self.device_obj, non_blocking=True)
        attributes = metadata[:, 0]  # First column (of the metadata batch) contains the attribute

        batch_group_keys = labels * 2 + attributes  # Group keys of examples in the batch

        for key in self.group_keys:
            mask = (batch_group_keys == key)  # mask=1 for examples belonging to the group

            num_samples_for_group = mask.sum().item()
            num_incorrect_preds_for_group = is_incorrect_preds[mask].sum().item()  # Incorrect samples in the group

            self.group_num_samples[key] += num_samples_for_group
            self.group_num_incorrect_preds[key] += num_incorrect_preds_for_group  # update the group error count

    def compute_group_errors(self):
        assert self.is_error_computed is False, 'Group errors have already been computed for this dataset'

        group_errors = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}

        for key in self.group_keys:
            if self.group_num_samples[key] > 0:
                group_errors[key] = self.group_num_incorrect_preds[key] * 100 / self.group_num_samples[key]

        # print(f'group_num_samples: {self.group_num_samples}')
        # print(f'group_num_incorrect_preds: {group_num_incorrect_preds}')
        # print(f'group_errors: {group_errors}')

        self.is_error_computed = True

        return group_errors

    def convert_to_group_accuracies(self, group_errors):
        group_accuracies = {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0}

        for key, error in group_errors.items():
            if error >= 0:
                group_accuracies[key] = 100 - error

        return group_accuracies
