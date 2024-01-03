import json


def get_slots_pt(type_slot, filled_slot):
    var = ""
    not_found_filled_slot, not_found_type_slot = 0, 0
    with open("C:/Users/Francisco Pais/Desktop/multiwozPortugues-main/multiwozPortugues-main/schemaPT.json",
              encoding='utf-8') as json_file:
        data = json.load(json_file)
        for i in data.keys():
            if i == type_slot:
                not_found_type_slot = 1
                for j in data[i]:
                    for key in j:
                        if key == filled_slot:
                            var = j[key]
                            not_found_filled_slot = 1
    if not_found_type_slot == 0:
        var = "Not Found Type Slot"
        return var
    if not_found_filled_slot == 0 and not_found_type_slot == 1:
        var = "Not Found Filled Slot"
        return var
    return var