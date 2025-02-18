import os
import sys
import numpy as np
import util

# init
DATA_DIR = sys.argv[1]
TOP_K_PLOT = 10
changesets = util.Changesets(DATA_DIR)
top_k_table, index_to_rank, rank_to_name = util.get_rank_infos(DATA_DIR, "created_by")
name_to_color = util.get_unique_name_to_color_mapping(
    rank_to_name["changesets"][:TOP_K_PLOT],
    rank_to_name["edits"][:TOP_K_PLOT],
    rank_to_name["contributors"][:TOP_K_PLOT],
)
name_to_link = util.load_name_to_link("replace_rules_created_by.json")


device_type_labels = ["desktop editor", "mobile editor", "tools", "other/unspecified"]
tag_to_index = util.list_to_dict(util.load_index_to_tag(DATA_DIR, "created_by"))

name_to_info = util.load_json(os.path.join("src", "replace_rules_created_by.json"))
desktop_editors, mobile_editors, tools = [], [], []
for name, info in name_to_info.items():
    if name not in tag_to_index:
        print(f"{name} in 'replace_rules_created_by.json' but not in 'tag_to_index' (this is expected when working"
        " with a part of all changesets)")
        continue
    if "type" in info:
        if info["type"] == "desktop_editor":
            desktop_editors.append(tag_to_index[name])
        elif info["type"] == "mobile_editor":
            mobile_editors.append(tag_to_index[name])
        elif info["type"] == "tool":
            tools.append(tag_to_index[name])
        else:
            print(f"unknown type: {name['type']} at name {name}")


months, years = changesets.months, changesets.years
monthly_changesets = np.zeros((top_k_table, len(months)), dtype=np.int64)
monthly_edits = np.zeros((top_k_table, len(months)), dtype=np.int64)
monthly_edits_device_type = np.zeros((4, len(months)), dtype=np.int64)
monthly_contributor_sets = [[set() for _ in range(len(months))] for _ in range(top_k_table)]
monthly_contributor_device_type_sets = [[set() for _ in range(len(months))] for _ in range(4)]
contributor_first_edit_per_editing_software = [{} for _ in range(TOP_K_PLOT)]
contributor_first_editing_software = {}

# accumulate data
for csv_line in sys.stdin:
    changesets.update_data_with_csv_str(csv_line)
    month_index = changesets.month_index
    created_by_index = changesets.created_by_index
    user_index = changesets.user_index
    edits = changesets.edits

    if created_by_index is None:
        continue

    if user_index not in contributor_first_editing_software:
        contributor_first_editing_software[user_index] = (month_index, created_by_index)
    else:
        if month_index < contributor_first_editing_software[user_index][0]:
            contributor_first_editing_software[user_index] = (month_index, created_by_index)

    if created_by_index in index_to_rank["changesets"]:
        rank = index_to_rank["changesets"][created_by_index]
        monthly_changesets[rank, month_index] += 1

    if created_by_index in index_to_rank["edits"]:
        rank = index_to_rank["edits"][created_by_index]
        monthly_edits[rank, month_index] += edits

    if created_by_index in index_to_rank["contributors"]:
        rank = index_to_rank["contributors"][created_by_index]
        monthly_contributor_sets[rank][month_index].add(user_index)

        if rank < TOP_K_PLOT:
            if user_index not in contributor_first_edit_per_editing_software[rank]:
                contributor_first_edit_per_editing_software[rank][user_index] = month_index
            else:
                if month_index < contributor_first_edit_per_editing_software[rank][user_index]:
                    contributor_first_edit_per_editing_software[rank][user_index] = month_index

    if created_by_index in desktop_editors:
        monthly_edits_device_type[0, month_index] += edits
        monthly_contributor_device_type_sets[0][month_index].add(user_index)
    elif created_by_index in mobile_editors:
        monthly_edits_device_type[1, month_index] += edits
        monthly_contributor_device_type_sets[1][month_index].add(user_index)
    elif created_by_index in tools:
        monthly_edits_device_type[2, month_index] += edits
        monthly_contributor_device_type_sets[2][month_index].add(user_index)
    else:
        monthly_edits_device_type[3, month_index] += edits
        monthly_contributor_device_type_sets[3][month_index].add(user_index)


mo_contributor_first_editing_software = np.zeros((TOP_K_PLOT, len(months)), dtype=np.int32)
for month_index, created_by_id in contributor_first_editing_software.values():
    if created_by_id in index_to_rank["contributors"]:
        rank = index_to_rank["contributors"][created_by_id]
        if rank < TOP_K_PLOT:
            mo_contributor_first_editing_software[rank][month_index] += 1

mo_new_contributor_per_editing_software = np.zeros((TOP_K_PLOT, len(months)), dtype=np.int32)
for rank in range(TOP_K_PLOT):
    month_index, first_edit_count = np.unique(
        list(contributor_first_edit_per_editing_software[rank].values()), return_counts=True
    )
    mo_new_contributor_per_editing_software[rank][month_index] = first_edit_count

mo_co = util.set_to_length(monthly_contributor_sets)


# save plots
TOPIC = "Editing Software"
with util.add_questions(TOPIC) as add_question:

    add_question(
        "How many people are contributing per editing software each month?",
        "c229",
        util.get_multi_line_plot(
            "monthly contributor count per editing software",
            "contributors",
            months,
            mo_co[:TOP_K_PLOT],
            rank_to_name["contributors"][:TOP_K_PLOT],
        ),
        util.get_multi_line_plot(
            "monthly new contributor count per editing software",
            "contributors",
            months,
            mo_new_contributor_per_editing_software,
            rank_to_name["contributors"][:TOP_K_PLOT],
        ),
        util.get_table(
            "yearly contributor count per editing software",
            years,
            util.monthly_set_to_yearly_with_total(
                monthly_contributor_sets, years, changesets.month_index_to_year_index
            ),
            TOPIC,
            rank_to_name["contributors"],
            name_to_link,
        ),
    )

    add_question(
        "How popular is each editing software per month?",
        "158c",
        util.get_multi_line_plot(
            "percent of contributors that use each editing software per month",
            "%",
            months,
            util.get_percent(mo_co[:TOP_K_PLOT], changesets.monthly_contributors),
            rank_to_name["contributors"][:TOP_K_PLOT],
            percent=True,
        ),
    )

    add_question(
        "Which editing software is used for the first edit?",
        "7662",
        util.get_multi_line_plot(
            "monthly first editing software contributor count",
            "contributors",
            months,
            mo_contributor_first_editing_software,
            rank_to_name["contributors"][:TOP_K_PLOT],
        ),
    )

    add_question(
        "How many edits are added per editing software each month?",
        "eb30",
        util.get_multi_line_plot(
            "monthly edits count per editing software",
            "edits",
            months,
            monthly_edits[:TOP_K_PLOT],
            rank_to_name["edits"][:TOP_K_PLOT],
        ),
        util.get_table(
            "yearly edits count per editing software",
            years,
            util.monthly_to_yearly_with_total(monthly_edits, years, changesets.month_index_to_year_index),
            TOPIC,
            rank_to_name["edits"],
        ),
    )

    add_question(
        "What's the market share of edits per month?",
        "a008",
        util.get_multi_line_plot(
            "market share of edits per month",
            "%",
            months,
            util.get_percent(monthly_edits, changesets.monthly_edits)[:TOP_K_PLOT],
            rank_to_name["edits"][:TOP_K_PLOT],
            percent=True,
            on_top_of_each_other=True,
        ),
    )

    add_question(
        "What's the total amount of contributors, edits and changesets of editing software over time?",
        "6320",
        util.get_text_element(
            f"There are {len(util.load_index_to_tag(DATA_DIR, 'created_by')):,} different editing software names for"
            " the 'created_by' tag. That's quite a big number. However, most names are user or organization names,"
            " from people misunderstanding the tag.",
        ),
        util.get_multi_line_plot(
            "total contributor count of editing software",
            "contributors",
            months,
            util.set_cumsum(monthly_contributor_sets),
            rank_to_name["contributors"][:TOP_K_PLOT],
            colors=[name_to_color[name] for name in rank_to_name["contributors"][:TOP_K_PLOT]],
        ),
        util.get_multi_line_plot(
            "total edit count of editing software",
            "edits",
            months,
            util.cumsum(monthly_edits),
            rank_to_name["edits"][:TOP_K_PLOT],
            colors=[name_to_color[name] for name in rank_to_name["edits"][:TOP_K_PLOT]],
        ),
        util.get_multi_line_plot(
            "total changeset count of editing software",
            "changesets",
            months,
            util.cumsum(monthly_changesets),
            rank_to_name["changesets"][:TOP_K_PLOT],
            colors=[name_to_color[name] for name in rank_to_name["changesets"][:TOP_K_PLOT]],
        ),
    )

    add_question(
        "What kind of devices are used for mapping?",
        "8ba9",
        util.get_multi_line_plot(
            "monthly contributor count per device",
            "contributors",
            months,
            util.set_to_length(monthly_contributor_device_type_sets),
            device_type_labels,
        ),
        util.get_multi_line_plot(
            "monthly edit count per device", "edits", months, monthly_edits_device_type, device_type_labels
        ),
        util.get_multi_line_plot(
            "market share of edit per device",
            "%",
            months,
            util.get_percent(monthly_edits_device_type, changesets.monthly_edits),
            device_type_labels,
            percent=True,
        ),
    )
