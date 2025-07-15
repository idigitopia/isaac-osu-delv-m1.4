from collections import defaultdict

from prettytable import PrettyTable


def get_common_terms_repr(group_term_names, group_term_cfgs):
    # Map (name, repr(cfg)) -> set of groups
    term_to_groups = defaultdict(set)
    for group in group_term_names:
        for name, cfg in zip(group_term_names[group], group_term_cfgs[group]):
            key = (name, repr(cfg))
            term_to_groups[key].add(group)

    # Invert: frozenset(groups) -> [term names]
    group_sets_to_terms = defaultdict(list)
    for (name, _), groups in term_to_groups.items():
        if len(groups) >= 2:
            group_sets_to_terms[frozenset(groups)].append(name)

    # Print common terms
    msg = "Common Terms between groups:\n"
    if not group_sets_to_terms:
        msg += "  (None)\n"
    else:
        for group_set, term_names in group_sets_to_terms.items():
            table = PrettyTable()
            table.title = f"Groups: {', '.join(sorted(group_set))}"
            table.field_names = ["Term Name"]
            for name in term_names:
                table.add_row([name])
            msg += table.get_string() + "\n"

    return msg
