import json

listaAttraction, listaHotel, listaRestaurant, listaTaxi, listaTrain = [], [], [], [], []


def get_questions():
    with open("C:/Users/Francisco Pais/Desktop/multiwozPortugues-main/multiwozPortugues-main/questionsv2.json",
              encoding='utf-8') as json_file:
        data = json.load(json_file)
        for i in data.keys():
            if i == "attraction":
                for j in data[i]:
                    for key in j:
                        listaAttraction.append(j[key])
            if i == "hotel":
                for j in data[i]:
                    for key in j:
                        listaHotel.append(j[key])
            if i == "restaurant":
                for j in data[i]:
                    for key in j:
                        listaRestaurant.append(j[key])
            if i == "taxi":
                for j in data[i]:
                    for key in j:
                        listaTaxi.append(j[key])
            if i == "train":
                for j in data[i]:
                    for key in j:
                        listaTrain.append(j[key])
    return listaAttraction, listaHotel, listaRestaurant, listaTaxi, listaTrain


