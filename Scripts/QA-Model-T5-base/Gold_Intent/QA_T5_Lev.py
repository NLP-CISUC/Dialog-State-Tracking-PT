# -*- coding: utf-8 -*-

from get_questions import get_questions

from get_slots_pt import get_slots_pt

from transformers import pipeline

import csv

from Levenshtein import distance as lev

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import numpy as np

# Portuguese Model & tokenizer T5 Base QA
model_name = "pierreguillou/t5-base-qa-squad-v1.1-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# parameters
max_target_length = 32
num_beams = 1
early_stopping = True
output_scores = True
return_dict_in_generate = True

# Save the questions that will be used in the QA Model in the following lists
listaAttraction, listaHotel, listaRestaurant, listaTaxi, listaTrain = get_questions()

# Variables to evaluate the Model's performance in each Domain
# ATTRACTION
cont_attraction_notfilled, cont_attraction_certo, cont_attraction_errado, cont_attraction_filledincorrect, cont_attraction_area_certo, cont_attraction_area_errado, \
cont_attraction_name_certo, cont_attraction_name_errado, cont_attraction_type_certo, cont_attraction_type_errado = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# HOTEL
cont_hotel_notfilled, cont_hotel_certo, cont_hotel_errado, cont_hotel_filledincorrect, cont_hotel_area_certo, cont_hotel_area_errado, cont_hotel_bookday_certo, \
cont_hotel_bookday_errado, cont_hotel_bookpeople_certo, cont_hotel_bookpeople_errado, cont_hotel_bookstay_certo, cont_hotel_bookstay_errado, \
cont_hotel_internet_certo, cont_hotel_internet_errado, cont_hotel_name_certo, cont_hotel_name_errado, cont_hotel_parking_certo, cont_hotel_parking_errado, \
cont_hotel_pricerange_certo, cont_hotel_pricerange_errado, cont_hotel_stars_certo, cont_hotel_stars_errado, cont_hotel_type_certo, cont_hotel_type_errado = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# RESTAURANT
cont_restaurant_notfilled, cont_restaurant_certo, cont_restaurant_errado, cont_restaurant_filledincorrect, cont_restaurant_area_certo, cont_restaurant_area_errado, \
cont_restaurant_bookday_certo, cont_restaurant_bookday_errado, cont_restaurant_bookpeople_certo, cont_restaurant_bookpeople_errado, \
cont_restaurant_booktime_certo, cont_restaurant_booktime_errado, cont_restaurant_food_certo, cont_restaurant_food_errado, cont_restaurant_name_certo, \
cont_restaurant_name_errado, cont_restaurant_pricerange_certo, cont_restaurant_pricerange_errado = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# TAXI
cont_taxi_notfilled, cont_taxi_certo, cont_taxi_errado, cont_taxi_filledincorrect, cont_taxi_arriveby_certo, cont_taxi_arriveby_errado, cont_taxi_departure_certo, \
cont_taxi_departure_errado, cont_taxi_destination_certo, cont_taxi_destination_errado, cont_taxi_leaveat_certo, cont_taxi_leaveat_errado = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# TRAIN
cont_train_notfilled, cont_train_certo, cont_train_errado, cont_train_filledincorrect, cont_train_arriveby_certo, cont_train_arriveby_errado, cont_train_bookpeople_certo, \
cont_train_bookpeople_errado, cont_train_day_certo, cont_train_day_errado, cont_train_departure_certo, cont_train_departure_errado, cont_train_destination_certo, \
cont_train_destination_errado, cont_train_leaveat_certo, cont_train_leaveat_errado = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# JGA
cont_turns_correct_attraction, cont_turns_correct_hotel, cont_turns_correct_restaurant, cont_turns_correct_taxi, cont_turns_correct_train = 0, 0, 0, 0, 0
cont_turns_incorrect_attraction, cont_turns_incorrect_hotel, cont_turns_incorrect_restaurant, cont_turns_incorrect_taxi, cont_turns_incorrect_train = 0, 0, 0, 0, 0

with open("C:/Users/Francisco Pais/Desktop/multiwozPortugues-main/multiwozPortugues-main"
          "/Dialogues/dialogosCoimbra488.json",
          "r", encoding='utf-8') as f:
    for line in f:
        linha = line.strip()
        linha = linha.split(':')
        if len(linha) == 2:
            if linha[0] == "\"dialogue_id\"":
                # New Dialogue found - restart of slots of the domains filled
                at_area, at_name, at_type = 1, 1, 1
                ht_area, ht_bookday, ht_bookpeople, ht_bookstay, ht_internet, ht_name, ht_parking, ht_pricerange, ht_stars, ht_type = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                rt_area, rt_bookday, rt_bookpeople, rt_booktime, rt_food, rt_name, rt_pricerange = 1, 1, 1, 1, 1, 1, 1
                tx_arriveby, tx_departure, tx_destination, tx_leaveat = 1, 1, 1, 1
                tr_arriveby, tr_bookpeople, tr_day, tr_departure, tr_destination, tr_leaveat = 1, 1, 1, 1, 1, 1
                attraction_type, attraction_area = [], []
                attraction_name = ""
                attraction_area_2, attraction_name_2, attraction_type_2 = "", "", ""
                hotel_pricerange, hotel_bookstay, hotel_bookpeople, hotel_stars, hotel_parking, hotel_internet, hotel_area, hotel_type, hotel_bookday = [], [], [], [], [], [], [], [], []
                hotel_name = ""
                hotel_area_2, hotel_bookday_2, hotel_bookpeople_2, hotel_bookstay_2, hotel_internet_2, hotel_name_2, hotel_parking_2, hotel_pricerange_2, hotel_stars_2, hotel_type_2 = "", "", "", "", "", "", "", "", "", ""
                restaurant_pricerange, restaurant_area, restaurant_bookday, restaurant_bookpeople = [], [], [], []
                restaurant_food, restaurant_name, restaurant_booktime = "", "", ""
                restaurant_area_2, restaurant_bookday_2, restaurant_bookpeople_2, restaurant_booktime_2, restaurant_food_2, restaurant_name_2, restaurant_pricerange_2 = "", "", "", "", "", "", ""
                taxi_arriveby, taxi_departure, taxi_destination, taxi_leaveat = "", "", "", ""
                taxi_arriveby_2, taxi_departure_2, taxi_destination_2, taxi_leaveat_2 = "", "", "", ""
                train_day, train_departure, train_destination, train_bookpeople = [], [], [], []
                train_arriveby, train_leaveat = "", ""
                train_arriveby_2, train_bookpeople_2, train_day_2, train_departure_2, train_destination_2, train_leaveat_2 = "", "", "", "", "", ""
            active_intent = linha[0]
            active_intent = active_intent[1:]
            active_intent = active_intent[:-1]
            active_intent2 = linha[1]
            active_intent2 = active_intent2[1:]
            active_intent2 = active_intent2[1:]
            active_intent2 = active_intent2[:-1]
            active_intent2 = active_intent2[:-1]
            turn_id = active_intent
            if active_intent2.isnumeric():
                number = active_intent2
            if active_intent == "active_intent" and active_intent2 != "NONE":
                active_intent2 = active_intent2[5:]
                present_intent = active_intent2
                if active_intent2 == "attraction":
                    for line in f:
                        linha = line.strip()
                        linha = linha.split(':')
                        slot_values = linha[0]
                        if linha[0] == "}":
                            break
                        if slot_values == "\"slot_values\"":
                            for line in f:
                                linha = line.strip()
                                linha = linha.split(':')
                                if linha[0] == "}":
                                    break
                                if len(linha) == 2:
                                    if linha[1] == " [":
                                        # Filling in the respective slots associated with each intent
                                        if linha[0] == "\"attraction-area\"":
                                            for line in f:
                                                attraction_area.append(get_slots_pt("attraction-area", "centre"))
                                                attraction_area.append(get_slots_pt("attraction-area", "east"))
                                                attraction_area.append(get_slots_pt("attraction-area", "north"))
                                                attraction_area.append(get_slots_pt("attraction-area", "south"))
                                                attraction_area.append(get_slots_pt("attraction-area", "west"))
                                                attraction_area.append(get_slots_pt("attraction-area", "dontcare"))
                                                break
                                        if linha[0] == "\"attraction-name\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                attraction_name = linha[0]
                                                attraction_name = attraction_name[1:]
                                                attraction_name = attraction_name[:-1]
                                                break
                                        if linha[0] == "\"attraction-type\"":
                                            for line in f:
                                                attraction_type.append(get_slots_pt("attraction-type", "architecture"))
                                                attraction_type.append(get_slots_pt("attraction-type", "boat"))
                                                attraction_type.append(get_slots_pt("attraction-type", "cinema"))
                                                attraction_type.append(get_slots_pt("attraction-type", "college"))
                                                attraction_type.append(get_slots_pt("attraction-type", "concerthall"))
                                                attraction_type.append(get_slots_pt("attraction-type", "entertainment"))
                                                attraction_type.append(get_slots_pt("attraction-type", "museum"))
                                                attraction_type.append(
                                                    get_slots_pt("attraction-type", "multiple sports"))
                                                attraction_type.append(get_slots_pt("attraction-type", "nightclub"))
                                                attraction_type.append(get_slots_pt("attraction-type", "park"))
                                                attraction_type.append(get_slots_pt("attraction-type", "swimmingpool"))
                                                attraction_type.append(get_slots_pt("attraction-type", "theater"))
                                                attraction_type.append(get_slots_pt("attraction-type", "dontcare"))
                                                break
                if active_intent2 == "hotel":
                    for line in f:
                        linha = line.strip()
                        linha = linha.split(':')
                        slot_values = linha[0]
                        if linha[0] == "}":
                            break
                        if slot_values == "\"slot_values\"":
                            for line in f:
                                linha = line.strip()
                                linha = linha.split(':')
                                if linha[0] == "}":
                                    break
                                if len(linha) == 2:
                                    if linha[1] == " [":
                                        if linha[0] == "\"hotel-area\"":
                                            for line in f:
                                                hotel_area.append(get_slots_pt("hotel-area", "centre"))
                                                hotel_area.append(get_slots_pt("hotel-area", "east"))
                                                hotel_area.append(get_slots_pt("hotel-area", "north"))
                                                hotel_area.append(get_slots_pt("hotel-area", "south"))
                                                hotel_area.append(get_slots_pt("hotel-area", "west"))
                                                hotel_area.append(get_slots_pt("hotel-area", "dontcare"))
                                                break
                                        if linha[0] == "\"hotel-bookday\"":
                                            for line in f:
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "monday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "tuesday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "wednesday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "thursday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "friday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "saturday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "sunday"))
                                                hotel_bookday.append(get_slots_pt("hotel-bookday", "dontcare"))
                                                break
                                        if linha[0] == "\"hotel-bookpeople\"":
                                            for line in f:
                                                hotel_bookpeople.append("1")
                                                hotel_bookpeople.append("2")
                                                hotel_bookpeople.append("3")
                                                hotel_bookpeople.append("4")
                                                hotel_bookpeople.append("5")
                                                hotel_bookpeople.append("6")
                                                hotel_bookpeople.append("7")
                                                hotel_bookpeople.append("8")
                                                break
                                        if linha[0] == "\"hotel-bookstay\"":
                                            for line in f:
                                                hotel_bookstay.append("1")
                                                hotel_bookstay.append("2")
                                                hotel_bookstay.append("3")
                                                hotel_bookstay.append("4")
                                                hotel_bookstay.append("5")
                                                hotel_bookstay.append("6")
                                                hotel_bookstay.append("7")
                                                hotel_bookstay.append("8")
                                                break
                                        if linha[0] == "\"hotel-internet\"":
                                            for line in f:
                                                hotel_internet.append(get_slots_pt("hotel-internet", "free"))
                                                hotel_internet.append(get_slots_pt("hotel-internet", "no"))
                                                hotel_internet.append(get_slots_pt("hotel-internet", "yes"))
                                                hotel_internet.append(get_slots_pt("hotel-internet", "dontcare"))
                                                break
                                        if linha[0] == "\"hotel-name\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                hotel_name = linha[0]
                                                hotel_name = hotel_name[1:]
                                                hotel_name = hotel_name[:-1]
                                                if hotel_name == "dontcare":
                                                    hotel_name = get_slots_pt("hotel-name", hotel_name)
                                                # print("hotel_name - ", hotel_name)
                                                break
                                        if linha[0] == "\"hotel-parking\"":
                                            for line in f:
                                                hotel_parking.append(get_slots_pt("hotel-parking", "free"))
                                                hotel_parking.append(get_slots_pt("hotel-parking", "no"))
                                                hotel_parking.append(get_slots_pt("hotel-parking", "yes"))
                                                hotel_parking.append(get_slots_pt("hotel-parking", "dontcare"))
                                                break
                                        if linha[0] == "\"hotel-pricerange\"":
                                            for line in f:
                                                hotel_pricerange.append(get_slots_pt("hotel-pricerange", "expensive"))
                                                hotel_pricerange.append(get_slots_pt("hotel-pricerange", "cheap"))
                                                hotel_pricerange.append(get_slots_pt("hotel-pricerange", "moderate"))
                                                hotel_pricerange.append(get_slots_pt("hotel-pricerange", "dontcare"))
                                                break
                                        if linha[0] == "\"hotel-stars\"":
                                            for line in f:
                                                hotel_stars.append("0")
                                                hotel_stars.append("1")
                                                hotel_stars.append("2")
                                                hotel_stars.append("3")
                                                hotel_stars.append("4")
                                                hotel_stars.append("5")
                                                break
                                        if linha[0] == "\"hotel-type\"":
                                            for line in f:
                                                hotel_type.append(get_slots_pt("hotel-type", "guesthouse"))
                                                hotel_type.append(get_slots_pt("hotel-type", "hotel"))
                                                hotel_type.append(get_slots_pt("hotel-type", "dontcare"))
                                                break

                if active_intent2 == "restaurant":
                    for line in f:
                        linha = line.strip()
                        linha = linha.split(':')
                        slot_values = linha[0]
                        if linha[0] == "}":
                            break
                        if slot_values == "\"slot_values\"":
                            for line in f:
                                linha = line.strip()
                                linha = linha.split(':')
                                if linha[0] == "}":
                                    break
                                if len(linha) == 2:
                                    if linha[1] == " [":
                                        if linha[0] == "\"restaurant-area\"":
                                            for line in f:
                                                restaurant_area.append(get_slots_pt("restaurant-area", "centre"))
                                                restaurant_area.append(get_slots_pt("restaurant-area", "east"))
                                                restaurant_area.append(get_slots_pt("restaurant-area", "north"))
                                                restaurant_area.append(get_slots_pt("restaurant-area", "south"))
                                                restaurant_area.append(get_slots_pt("restaurant-area", "west"))
                                                restaurant_area.append(get_slots_pt("restaurant-area", "dontcare"))
                                                break
                                        if linha[0] == "\"restaurant-bookday\"":
                                            for line in f:
                                                restaurant_bookday.append(get_slots_pt("restaurant-bookday", "monday"))
                                                restaurant_bookday.append(get_slots_pt("restaurant-bookday", "tuesday"))
                                                restaurant_bookday.append(
                                                    get_slots_pt("restaurant-bookday", "wednesday"))
                                                restaurant_bookday.append(
                                                    get_slots_pt("restaurant-bookday", "thursday"))
                                                restaurant_bookday.append(get_slots_pt("restaurant-bookday", "friday"))
                                                restaurant_bookday.append(
                                                    get_slots_pt("restaurant-bookday", "saturday"))
                                                restaurant_bookday.append(get_slots_pt("restaurant-bookday", "sunday"))
                                                restaurant_bookday.append(
                                                    get_slots_pt("restaurant-bookday", "dontcare"))
                                                break
                                        if linha[0] == "\"restaurant-bookpeople\"":
                                            for line in f:
                                                restaurant_bookpeople.append("1")
                                                restaurant_bookpeople.append("2")
                                                restaurant_bookpeople.append("3")
                                                restaurant_bookpeople.append("4")
                                                restaurant_bookpeople.append("5")
                                                restaurant_bookpeople.append("6")
                                                restaurant_bookpeople.append("7")
                                                restaurant_bookpeople.append("8")
                                                break
                                        if linha[0] == "\"restaurant-booktime\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                if len(linha) == 2:
                                                    restaurant_booktime = linha[0] + ":" + linha[1]
                                                    restaurant_booktime = restaurant_booktime[1:]
                                                    restaurant_booktime = restaurant_booktime[:-1]
                                                else:
                                                    restaurant_booktime = linha[0]
                                                    restaurant_booktime = restaurant_booktime[1:]
                                                    restaurant_booktime = restaurant_booktime[:-1]
                                                break
                                        if linha[0] == "\"restaurant-food\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                restaurant_food = linha[0]
                                                restaurant_food = restaurant_food[1:]
                                                restaurant_food = restaurant_food[:-1]
                                                restaurant_food = get_slots_pt("restaurant-food", restaurant_food)
                                                break
                                        if linha[0] == "\"restaurant-name\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                restaurant_name = linha[0]
                                                restaurant_name = restaurant_name[1:]
                                                restaurant_name = restaurant_name[:-1]
                                                if restaurant_name == "dontcare":
                                                    restaurant_name = get_slots_pt("restaurant-name", restaurant_name)
                                                break
                                        if linha[0] == "\"restaurant-pricerange\"":
                                            for line in f:
                                                restaurant_pricerange.append(
                                                    get_slots_pt("restaurant-pricerange", "expensive"))
                                                restaurant_pricerange.append(
                                                    get_slots_pt("restaurant-pricerange", "cheap"))
                                                restaurant_pricerange.append(
                                                    get_slots_pt("restaurant-pricerange", "moderate"))
                                                restaurant_pricerange.append(
                                                    get_slots_pt("restaurant-pricerange", "dontcare"))
                                                break

                if active_intent2 == "taxi":
                    for line in f:
                        linha = line.strip()
                        linha = linha.split(':')
                        slot_values = linha[0]
                        if linha[0] == "}":
                            break
                        if slot_values == "\"slot_values\"":
                            for line in f:
                                linha = line.strip()
                                linha = linha.split(':')
                                if linha[0] == "}":
                                    break
                                if len(linha) == 2:
                                    if linha[1] == " [":
                                        if linha[0] == "\"taxi-arriveby\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                if len(linha) == 2:
                                                    taxi_arriveby = linha[0] + ":" + linha[1]
                                                    taxi_arriveby = taxi_arriveby[1:]
                                                    taxi_arriveby = taxi_arriveby[:-1]
                                                else:
                                                    taxi_arriveby = linha[0]
                                                    taxi_arriveby = taxi_arriveby[1:]
                                                    taxi_arriveby = taxi_arriveby[:-1]
                                                break
                                        if linha[0] == "\"taxi-bookpeople\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                taxi_bookpeople = linha[0]
                                                taxi_bookpeople = taxi_bookpeople[1:]
                                                taxi_bookpeople = taxi_bookpeople[:-1]
                                                break
                                        if linha[0] == "\"taxi-day\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                taxi_day = linha[0]
                                                taxi_day = taxi_day[1:]
                                                taxi_day = taxi_day[:-1]
                                                taxi_day = get_slots_pt("taxi-day", taxi_day)
                                                break
                                        if linha[0] == "\"taxi-departure\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                taxi_departure = linha[0]
                                                taxi_departure = taxi_departure[1:]
                                                taxi_departure = taxi_departure[:-1]
                                                break
                                        if linha[0] == "\"taxi-destination\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                taxi_destination = linha[0]
                                                taxi_destination = taxi_destination[1:]
                                                taxi_destination = taxi_destination[:-1]
                                                break
                                        if linha[0] == "\"taxi-leaveat\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                if len(linha) == 2:
                                                    taxi_leaveat = linha[0] + ":" + linha[1]
                                                    taxi_leaveat = taxi_leaveat[1:]
                                                    taxi_leaveat = taxi_leaveat[:-1]
                                                else:
                                                    taxi_leaveat = linha[0]
                                                    taxi_leaveat = taxi_leaveat[1:]
                                                    taxi_leaveat = taxi_leaveat[:-1]
                                                if taxi_leaveat == "dontcare":
                                                    taxi_leaveat = get_slots_pt("taxi-leaveat", taxi_leaveat)
                                                break

                if active_intent2 == "train":
                    for line in f:
                        linha = line.strip()
                        linha = linha.split(':')
                        slot_values = linha[0]
                        if linha[0] == "}":
                            break
                        if slot_values == "\"slot_values\"":
                            for line in f:
                                linha = line.strip()
                                linha = linha.split(':')
                                if linha[0] == "}":
                                    break
                                if len(linha) == 2:
                                    if linha[1] == " [":
                                        if linha[0] == "\"train-arriveby\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                if len(linha) == 2:
                                                    train_arriveby = linha[0] + ":" + linha[1]
                                                    train_arriveby = train_arriveby[1:]
                                                    train_arriveby = train_arriveby[:-1]
                                                else:
                                                    train_arriveby = linha[0]
                                                    train_arriveby = train_arriveby[1:]
                                                    train_arriveby = train_arriveby[:-1]
                                                break
                                        if linha[0] == "\"train-bookpeople\"":
                                            for line in f:
                                                train_bookpeople.append("1")
                                                train_bookpeople.append("2")
                                                train_bookpeople.append("3")
                                                train_bookpeople.append("4")
                                                train_bookpeople.append("5")
                                                train_bookpeople.append("6")
                                                train_bookpeople.append("7")
                                                train_bookpeople.append("8")
                                                train_bookpeople.append("9")
                                                train_bookpeople.append("10")
                                                train_bookpeople.append("11")
                                                train_bookpeople.append("12")
                                                train_bookpeople.append("13")
                                                train_bookpeople.append("14")
                                                train_bookpeople.append("15")
                                                break
                                        if linha[0] == "\"train-day\"":
                                            for line in f:
                                                train_day.append(get_slots_pt("train-day", "monday"))
                                                train_day.append(get_slots_pt("train-day", "tuesday"))
                                                train_day.append(get_slots_pt("train-day", "wednesday"))
                                                train_day.append(get_slots_pt("train-day", "thursday"))
                                                train_day.append(get_slots_pt("train-day", "friday"))
                                                train_day.append(get_slots_pt("train-day", "saturday"))
                                                train_day.append(get_slots_pt("train-day", "sunday"))
                                                train_day.append(get_slots_pt("train-day", "dontcare"))
                                                break
                                        if linha[0] == "\"train-departure\"":
                                            for line in f:
                                                train_departure.append("Coimbra")
                                                train_departure.append("Aveiro")
                                                train_departure.append("Fontela")
                                                train_departure.append("Figueira da Foz")
                                                train_departure.append("Oliveira do Bairro")
                                                train_departure.append("Lisboa Oriente")
                                                train_departure.append("Curia")
                                                train_departure.append("Entroncamento")
                                                train_departure.append("Mealhada")
                                                train_departure.append("Pereira")
                                                train_departure.append("Souselas")
                                                train_departure.append("Fátima")
                                                train_departure.append("Quintas")
                                                train_departure.append("dontcare")
                                                break
                                        if linha[0] == "\"train-destination\"":
                                            for line in f:
                                                train_destination.append("Coimbra")
                                                train_destination.append("Aveiro")
                                                train_destination.append("Fontela")
                                                train_destination.append("Figueira da Foz")
                                                train_destination.append("Oliveira do Bairro")
                                                train_destination.append("Lisboa Oriente")
                                                train_destination.append("Curia")
                                                train_destination.append("Entroncamento")
                                                train_destination.append("Mealhada")
                                                train_destination.append("Pereira")
                                                train_destination.append("Souselas")
                                                train_destination.append("Fátima")
                                                train_destination.append("Quintas")
                                                train_destination.append("dontcare")
                                                break
                                        if linha[0] == "\"train-leaveat\"":
                                            for line in f:
                                                linha = line.strip()
                                                linha = linha.split(':')
                                                if len(linha) == 2:
                                                    train_leaveat = linha[0] + ":" + linha[1]
                                                    train_leaveat = train_leaveat[1:]
                                                    train_leaveat = train_leaveat[:-1]
                                                else:
                                                    train_leaveat = linha[0]
                                                    train_leaveat = train_leaveat[1:]
                                                    train_leaveat = train_leaveat[:-1]
                                                break
            verify_attraction, verify_hotel, verify_restaurant, verify_taxi, verify_train = 2, 2, 2, 2, 2
            # Search only for User Utterances
            if turn_id == "turn_id" and int(number) % 2 == 0:
                for line in f:
                    linha = line.strip()
                    linha = linha.split('"')
                    # Reached the end of the utterance
                    if linha[0] == "},":
                        break
                    utterance = linha[3]
                    # Filling of Slots by the QA Model above a certain value for the 5 Domains
                    if present_intent == "attraction": # if 1 == 1:
                        for i in range(len(listaAttraction)):
                            input_text = 'question:' + ' ' + listaAttraction[i] + ' ' + 'context:' + ' ' + utterance
                            inputs = tokenizer(input_text, return_tensors="pt")
                            outputs = model.generate(inputs["input_ids"],
                                                     max_length=max_target_length,
                                                     num_beams=num_beams,
                                                     early_stopping=early_stopping,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate
                                                     )
                            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,normalize_logits=True)
                            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                            generated_tokens = outputs.sequences[:, input_length:]
                            score_total = 0
                            x = 0
                            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                                x = x + 1
                                score_total = score_total + np.exp(score.numpy())
                            score_total = score_total - np.exp(score.numpy())
                            score = round(score_total / (x - 1), 2)
                            correct = 0
                            if i == 0 and score > 0.90:
                                attraction_area_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                for y in range(len(attraction_area)):
                                    if lev(attraction_area[y], attraction_area_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_attraction = 1
                                    cont_attraction_certo = cont_attraction_certo + 1
                                    cont_attraction_area_certo = cont_attraction_area_certo + 1
                                elif len(attraction_area) == 0:
                                    verify_attraction = 0
                                    cont_attraction_filledincorrect = cont_attraction_filledincorrect + 1
                                else:
                                    verify_attraction = 0
                                    cont_attraction_errado = cont_attraction_errado + 1
                                    cont_attraction_area_errado = cont_attraction_area_errado + 1
                            if i == 0:
                                if len(attraction_area) != 0 and attraction_area_2 == "" and at_area == 1:
                                    verify_attraction = 0
                                    at_area = 0
                                    cont_attraction_notfilled = cont_attraction_notfilled + 1
                            if i == 1 and score > 0.95:
                                attraction_name_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(attraction_name, attraction_name_2) <= 5:
                                    verify_attraction = 1
                                    cont_attraction_certo = cont_attraction_certo + 1
                                    cont_attraction_name_certo = cont_attraction_name_certo + 1
                                elif attraction_name == "":
                                    verify_attraction = 0
                                    cont_attraction_filledincorrect = cont_attraction_filledincorrect + 1
                                else:
                                    verify_attraction = 0
                                    cont_attraction_errado = cont_attraction_errado + 1
                                    cont_attraction_name_errado = cont_attraction_name_errado + 1
                            if i == 1:
                                if attraction_name != "" and attraction_name_2 == "" and at_name == 1:
                                    verify_attraction = 0
                                    at_name = 0
                                    cont_attraction_notfilled = cont_attraction_notfilled + 1
                            if i == 2 and score > 0.90:
                                attraction_type_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(attraction_type)):
                                    if lev(attraction_type[y], attraction_type_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_attraction = 1
                                    cont_attraction_certo = cont_attraction_certo + 1
                                    cont_attraction_type_certo = cont_attraction_type_certo + 1
                                elif len(attraction_type) == 0:
                                    verify_attraction = 0
                                    cont_attraction_filledincorrect = cont_attraction_filledincorrect + 1
                                else:
                                    verify_attraction = 0
                                    cont_attraction_errado = cont_attraction_errado + 1
                                    cont_attraction_type_errado = cont_attraction_type_errado + 1
                            if i == 2:
                                if len(attraction_type) != 0 and attraction_type_2 == "" and at_type == 1:
                                    verify_attraction = 0
                                    at_type = 0
                                    cont_attraction_notfilled = cont_attraction_notfilled + 1
                        # after the evaluation per turn it is checked if the model got all the slots right for that turn -> JGA
                        if verify_attraction == 1:
                            cont_turns_correct_attraction = cont_turns_correct_attraction + 1
                        if verify_attraction == 0:
                            cont_turns_incorrect_attraction = cont_turns_incorrect_attraction + 1
                    if present_intent == "hotel": # if 1 == 1:
                        for i in range(len(listaHotel)):
                            input_text = 'question:' + ' ' + listaHotel[i] + ' ' + 'context:' + ' ' + utterance
                            inputs = tokenizer(input_text, return_tensors="pt")
                            outputs = model.generate(inputs["input_ids"],
                                                     max_length=max_target_length,
                                                     num_beams=num_beams,
                                                     early_stopping=early_stopping,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate
                                                     )
                            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,normalize_logits=True)
                            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                            generated_tokens = outputs.sequences[:, input_length:]
                            score_total = 0
                            x = 0
                            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                                x = x + 1
                                score_total = score_total + np.exp(score.numpy())
                            score_total = score_total - np.exp(score.numpy())
                            score = round(score_total / (x - 1), 2)
                            correct = 0
                            if i == 0 and score > 0.95:
                                hotel_area_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_area)):
                                    if lev(hotel_area[y], hotel_area_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_area_certo = cont_hotel_area_certo + 1
                                elif len(hotel_area) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_area_errado = cont_hotel_area_errado + 1
                            if i == 0:
                                if len(hotel_area) != 0 and hotel_area_2 == "" and ht_area == 1:
                                    verify_hotel = 0
                                    ht_area = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 1 and score > 0.90:
                                hotel_bookday_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_bookday)):
                                    if lev(hotel_bookday[y], hotel_bookday_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_bookday_certo = cont_hotel_bookday_certo + 1
                                elif len(hotel_bookday) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_bookday_errado = cont_hotel_bookday_errado + 1
                            if i == 1:
                                if len(hotel_bookday) != 0 and hotel_bookday_2 == "" and ht_bookday == 1:
                                    verify_hotel = 0
                                    ht_bookday = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 2 and score > 0.80:
                                hotel_bookpeople_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_bookpeople)):
                                    if hotel_bookpeople[y] == hotel_bookpeople_2:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_bookpeople_certo = cont_hotel_bookpeople_certo + 1
                                elif len(hotel_bookpeople) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_bookpeople_errado = cont_hotel_bookpeople_errado + 1
                            if i == 2:
                                if len(hotel_bookpeople) != 0 and hotel_bookpeople_2 == "" and ht_bookpeople == 1:
                                    verify_hotel = 0
                                    ht_bookpeople = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 3 and score > 0.95:
                                hotel_bookstay_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_bookstay)):
                                    if hotel_bookstay[y] == hotel_bookstay_2:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_bookstay_certo = cont_hotel_bookstay_certo + 1
                                elif len(hotel_bookstay) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_bookstay_errado = cont_hotel_bookstay_errado + 1
                            if i == 3:
                                if len(hotel_bookstay) != 0 and hotel_bookstay_2 == "" and ht_bookstay == 1:
                                    verify_hotel = 0
                                    ht_bookstay = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 4 and score > 0.95:
                                hotel_internet_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_internet)):
                                    if lev(hotel_internet[y], hotel_internet_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_internet_certo = cont_hotel_internet_certo + 1
                                elif len(hotel_internet) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_internet_errado = cont_hotel_internet_errado + 1
                            if i == 4:
                                if len(hotel_internet) != 0 and hotel_internet_2 == "" and ht_internet == 1:
                                    verify_hotel = 0
                                    ht_internet = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 5 and score > 0.95:
                                hotel_name_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(hotel_name, hotel_name_2) <= 5:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_name_certo = cont_hotel_name_certo + 1
                                elif hotel_name == "":
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_name_errado = cont_hotel_name_errado + 1
                            if i == 5:
                                if hotel_name != "" and hotel_name_2 == "" and ht_name == 1:
                                    verify_hotel = 0
                                    ht_name = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 6 and score > 0.95:
                                hotel_parking_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_parking)):
                                    if lev(hotel_parking[y], hotel_parking_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_parking_certo = cont_hotel_parking_certo + 1
                                elif len(hotel_parking) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_parking_errado = cont_hotel_parking_errado + 1
                            if i == 6:
                                if len(hotel_parking) != 0 and hotel_parking_2 == "" and ht_parking == 1:
                                    verify_hotel = 0
                                    ht_parking = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 7 and score > 0.95:
                                hotel_pricerange_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_pricerange)):
                                    if lev(hotel_pricerange[y], hotel_pricerange_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_pricerange_certo = cont_hotel_pricerange_certo + 1
                                elif len(hotel_pricerange) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_pricerange_errado = cont_hotel_pricerange_errado + 1
                            if i == 7:
                                if len(hotel_pricerange) != 0 and hotel_pricerange_2 == "" and ht_pricerange == 1:
                                    verify_hotel = 0
                                    ht_pricerange = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 8 and score > 0.95:
                                hotel_stars_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_stars)):
                                    if hotel_stars[y] == hotel_stars_2:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_stars_certo = cont_hotel_stars_certo + 1
                                elif len(hotel_stars) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_stars_errado = cont_hotel_stars_errado + 1
                            if i == 8:
                                if len(hotel_stars) != 0 and hotel_stars_2 == "" and ht_stars == 1:
                                    verify_hotel = 0
                                    ht_stars = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                            if i == 9 and score > 0.95:
                                hotel_type_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(hotel_type)):
                                    if lev(hotel_type[y], hotel_type_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_hotel = 1
                                    cont_hotel_certo = cont_hotel_certo + 1
                                    cont_hotel_type_certo = cont_hotel_type_certo + 1
                                elif len(hotel_type) == 0:
                                    verify_hotel = 0
                                    cont_hotel_filledincorrect = cont_hotel_filledincorrect + 1
                                else:
                                    verify_hotel = 0
                                    cont_hotel_errado = cont_hotel_errado + 1
                                    cont_hotel_type_errado = cont_hotel_type_errado + 1
                            if i == 9:
                                if len(hotel_type) != 0 and hotel_type_2 == "" and ht_type == 1:
                                    verify_hotel = 0
                                    ht_type = 0
                                    cont_hotel_notfilled = cont_hotel_notfilled + 1
                        if verify_hotel == 1:
                            cont_turns_correct_hotel = cont_turns_correct_hotel + 1
                        if verify_hotel == 0:
                            cont_turns_incorrect_hotel = cont_turns_incorrect_hotel + 1
                    if present_intent == "restaurant": # if 1 == 1:
                        for i in range(len(listaRestaurant)):
                            input_text = 'question:' + ' ' + listaRestaurant[i] + ' ' + 'context:' + ' ' + utterance
                            inputs = tokenizer(input_text, return_tensors="pt")
                            outputs = model.generate(inputs["input_ids"],
                                                     max_length=max_target_length,
                                                     num_beams=num_beams,
                                                     early_stopping=early_stopping,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate
                                                     )
                            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,normalize_logits=True)
                            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                            generated_tokens = outputs.sequences[:, input_length:]
                            score_total = 0
                            x = 0
                            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                                x = x + 1
                                score_total = score_total + np.exp(score.numpy())
                            score_total = score_total - np.exp(score.numpy())
                            score = round(score_total / (x - 1), 2)
                            correct = 0
                            if i == 0 and score > 0.95:
                                restaurant_area_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(restaurant_area)):
                                    if lev(restaurant_area[y], restaurant_area_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_area_certo = cont_restaurant_area_certo + 1
                                elif len(restaurant_area) == 0:
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_area_errado = cont_restaurant_area_errado + 1
                            if i == 0:
                                if len(restaurant_area) != 0 and restaurant_area_2 == "" and rt_area == 1:
                                    verify_restaurant = 0
                                    rt_area = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 1 and score > 0.80:
                                restaurant_bookday_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(restaurant_bookday)):
                                    if lev(restaurant_bookday[y], restaurant_bookday_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_bookday_certo = cont_restaurant_bookday_certo + 1
                                elif len(restaurant_bookday) == 0:
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_bookday_errado = cont_restaurant_bookday_errado + 1
                            if i == 1:
                                if len(restaurant_bookday) != 0 and restaurant_bookday_2 == "" and rt_bookday == 1:
                                    verify_restaurant = 0
                                    rt_bookday = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 2 and score > 0.95:
                                restaurant_bookpeople_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(restaurant_bookpeople)):
                                    if restaurant_bookpeople[y] == restaurant_bookpeople_2:
                                        correct = 1
                                if correct == 1:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_bookpeople_certo = cont_restaurant_bookpeople_certo + 1
                                elif len(restaurant_bookpeople) == 0:
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_bookpeople_errado = cont_restaurant_bookpeople_errado + 1
                            if i == 2:
                                if len(restaurant_bookpeople) != 0 and restaurant_bookpeople_2 == "" and rt_bookpeople == 1:
                                    verify_restaurant = 0
                                    rt_bookpeople = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 3 and score > 0.95:
                                restaurant_booktime_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if restaurant_booktime == restaurant_booktime_2:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_booktime_certo = cont_restaurant_booktime_certo + 1
                                elif restaurant_booktime == "":
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_booktime_errado = cont_restaurant_booktime_errado + 1
                            if i == 3:
                                if restaurant_booktime != "" and restaurant_booktime_2 == "" and rt_booktime == 1:
                                    verify_restaurant = 0
                                    rt_booktime = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 4 and score > 0.90:
                                restaurant_food_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(restaurant_food, restaurant_food_2) <= 5:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_food_certo = cont_restaurant_food_certo + 1
                                elif restaurant_food == "":
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_food_errado = cont_restaurant_food_errado + 1
                            if i == 4:
                                if restaurant_food != "" and restaurant_food_2 == "" and rt_food == 1:
                                    verify_restaurant = 0
                                    rt_food = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 5 and score > 0.95:
                                restaurant_name_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(restaurant_name, restaurant_name_2) <= 5:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_name_certo = cont_restaurant_name_certo + 1
                                elif restaurant_name == "":
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_name_errado = cont_restaurant_name_errado + 1
                            if i == 5:
                                if restaurant_name != "" and restaurant_name_2 == "" and rt_name == 1:
                                    verify_restaurant = 0
                                    rt_name = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                            if i == 6 and score > 0.80:
                                restaurant_pricerange_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(restaurant_pricerange)):
                                    if lev(restaurant_pricerange[y], restaurant_pricerange_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_restaurant = 1
                                    cont_restaurant_certo = cont_restaurant_certo + 1
                                    cont_restaurant_pricerange_certo = cont_restaurant_pricerange_certo + 1
                                elif len(restaurant_pricerange) == 0:
                                    verify_restaurant = 0
                                    cont_restaurant_filledincorrect = cont_restaurant_filledincorrect + 1
                                else:
                                    verify_restaurant = 0
                                    cont_restaurant_errado = cont_restaurant_errado + 1
                                    cont_restaurant_pricerange_errado = cont_restaurant_pricerange_errado + 1
                            if i == 6:
                                if len(restaurant_pricerange) != 0 and restaurant_pricerange_2 == "" and rt_pricerange == 1:
                                    verify_restaurant = 0
                                    rt_pricerange = 0
                                    cont_restaurant_notfilled = cont_restaurant_notfilled + 1
                        if verify_restaurant == 1:
                            cont_turns_correct_restaurant = cont_turns_correct_restaurant + 1
                        if verify_restaurant == 0:
                            cont_turns_incorrect_restaurant = cont_turns_incorrect_restaurant + 1
                    if present_intent == "taxi": # if 1 == 1:
                        for i in range(len(listaTaxi)):
                            input_text = 'question:' + ' ' + listaTaxi[i] + ' ' + 'context:' + ' ' + utterance
                            inputs = tokenizer(input_text, return_tensors="pt")
                            outputs = model.generate(inputs["input_ids"],
                                                     max_length=max_target_length,
                                                     num_beams=num_beams,
                                                     early_stopping=early_stopping,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate
                                                     )
                            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,normalize_logits=True)
                            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                            generated_tokens = outputs.sequences[:, input_length:]
                            score_total = 0
                            x = 0
                            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                                x = x + 1
                                score_total = score_total + np.exp(score.numpy())
                            score_total = score_total - np.exp(score.numpy())
                            score = round(score_total / (x - 1), 2)
                            correct = 0
                            if i == 0 and score > 0.95:
                                taxi_arriveby_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if taxi_arriveby == taxi_arriveby_2:
                                    verify_taxi = 1
                                    cont_taxi_certo = cont_taxi_certo + 1
                                    cont_taxi_arriveby_certo = cont_taxi_arriveby_certo + 1
                                elif taxi_arriveby == "":
                                    verify_taxi = 0
                                    cont_taxi_filledincorrect = cont_taxi_filledincorrect + 1
                                else:
                                    verify_taxi = 0
                                    cont_taxi_errado = cont_taxi_errado + 1
                                    cont_taxi_arriveby_errado = cont_taxi_arriveby_errado + 1
                            if i == 0:
                                if taxi_arriveby != "" and taxi_arriveby_2 == "" and tx_arriveby == 1:
                                    verify_taxi = 0
                                    tx_arriveby = 0
                                    cont_taxi_notfilled = cont_taxi_notfilled + 1
                            if i == 1 and score > 0.95:
                                taxi_departure_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(taxi_departure, taxi_departure_2) <= 5:
                                    verify_taxi = 1
                                    cont_taxi_certo = cont_taxi_certo + 1
                                    cont_taxi_departure_certo = cont_taxi_departure_certo + 1
                                elif taxi_departure == "":
                                    verify_taxi = 0
                                    cont_taxi_filledincorrect = cont_taxi_filledincorrect + 1
                                else:
                                    verify_taxi = 0
                                    cont_taxi_errado = cont_taxi_errado + 1
                                    cont_taxi_departure_errado = cont_taxi_departure_errado + 1
                            if i == 1:
                                if taxi_departure != "" and taxi_departure_2 == "" and tx_departure == 1:
                                    verify_taxi = 0
                                    tx_departure = 0
                                    cont_taxi_notfilled = cont_taxi_notfilled + 1
                            if i == 2 and score > 0.95:
                                taxi_destination_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(taxi_destination, taxi_destination_2) <= 5:
                                    verify_taxi = 1
                                    cont_taxi_certo = cont_taxi_certo + 1
                                    cont_taxi_destination_certo = cont_taxi_destination_certo + 1
                                elif taxi_destination == "":
                                    verify_taxi = 0
                                    cont_taxi_filledincorrect = cont_taxi_filledincorrect + 1
                                else:
                                    verify_taxi = 0
                                    cont_taxi_errado = cont_taxi_errado + 1
                                    cont_taxi_destination_errado = cont_taxi_destination_errado + 1
                            if i == 2:
                                if taxi_destination != "" and taxi_destination_2 == "" and tx_destination == 1:
                                    verify_taxi = 0
                                    tx_destination = 0
                                    cont_taxi_notfilled = cont_taxi_notfilled + 1
                            if i == 3 and score > 0.95:
                                taxi_leaveat_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if taxi_leaveat == taxi_leaveat_2:
                                    verify_taxi = 1
                                    cont_taxi_certo = cont_taxi_certo + 1
                                    cont_taxi_leaveat_certo = cont_taxi_leaveat_certo + 1
                                elif taxi_leaveat == "":
                                    verify_taxi = 0
                                    cont_taxi_filledincorrect = cont_taxi_filledincorrect + 1
                                else:
                                    verify_taxi = 0
                                    cont_taxi_errado = cont_taxi_errado + 1
                                    cont_taxi_leaveat_errado = cont_taxi_leaveat_errado + 1
                            if i == 3:
                                if taxi_leaveat != "" and taxi_leaveat_2 == "" and tx_leaveat == 1:
                                    verify_taxi = 0
                                    tx_leaveat = 0
                                    cont_taxi_notfilled = cont_taxi_notfilled + 1
                        if verify_taxi == 1:
                            cont_turns_correct_taxi = cont_turns_correct_taxi + 1
                        if verify_taxi == 0:
                            cont_turns_incorrect_taxi = cont_turns_incorrect_taxi + 1
                    if present_intent == "train": # if 1 == 1:
                        for i in range(len(listaTrain)):
                            input_text = 'question:' + ' ' + listaTrain[i] + ' ' + 'context:' + ' ' + utterance
                            inputs = tokenizer(input_text, return_tensors="pt")
                            outputs = model.generate(inputs["input_ids"],
                                                     max_length=max_target_length,
                                                     num_beams=num_beams,
                                                     early_stopping=early_stopping,
                                                     output_scores=output_scores,
                                                     return_dict_in_generate=return_dict_in_generate
                                                     )
                            transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores,normalize_logits=True)
                            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                            generated_tokens = outputs.sequences[:, input_length:]
                            score_total = 0
                            x = 0
                            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                                x = x + 1
                                score_total = score_total + np.exp(score.numpy())
                            score_total = score_total - np.exp(score.numpy())
                            score = round(score_total / (x - 1), 2)
                            correct = 0
                            if i == 0 and score > 0.95:
                                train_arriveby_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if train_arriveby == train_arriveby_2:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_arriveby_certo = cont_train_arriveby_certo + 1
                                elif train_arriveby == "":
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_arriveby_errado = cont_train_arriveby_errado + 1
                            if i == 0:
                                if train_arriveby != "" and train_arriveby_2 == "" and tr_arriveby == 1:
                                    verify_train = 0
                                    tr_arriveby = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                            if i == 1 and score > 0.95:
                                train_bookpeople_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(train_bookpeople)):
                                    if train_bookpeople[y] == train_bookpeople_2:
                                        correct = 1
                                if correct == 1:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_bookpeople_certo = cont_train_bookpeople_certo + 1
                                elif len(train_bookpeople) == 0:
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_bookpeople_errado = cont_train_bookpeople_errado + 1
                            if i == 1:
                                if len(train_bookpeople) != 0 and train_bookpeople_2 == "" and tr_bookpeople == 1:
                                    verify_train = 0
                                    tr_bookpeople = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                            if i == 2 and score > 0.95:
                                train_day_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(train_day)):
                                    if lev(train_day[y], train_day_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_day_certo = cont_train_day_certo + 1
                                elif len(train_day) == 0:
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_day_errado = cont_train_day_errado + 1
                            if i == 2:
                                if len(train_day) != 0 and train_day_2 == "" and tr_day == 1:
                                    verify_train = 0
                                    tr_day = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                            if i == 3 and score > 0.95:
                                train_departure_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(train_departure)):
                                    if lev(train_departure[y], train_departure_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_departure_certo = cont_train_departure_certo + 1
                                elif len(train_departure) == 0:
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_departure_errado = cont_train_departure_errado + 1
                            if i == 3:
                                if len(train_departure) != 0 and train_departure_2 == "" and tr_departure == 1:
                                    verify_train = 0
                                    tr_departure = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                            if i == 4 and score > 0.95:
                                train_destination_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                for y in range(len(train_destination)):
                                    if lev(train_destination[y], train_destination_2) <= 5:
                                        correct = 1
                                if correct == 1:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_destination_certo = cont_train_destination_certo + 1
                                elif len(train_destination) == 0:
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_destination_errado = cont_train_destination_errado + 1
                            if i == 4:
                                if len(train_destination) != 0 and train_destination_2 == "" and tr_destination == 1:
                                    verify_train = 0
                                    tr_destination = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                            if i == 5 and score > 0.95:
                                train_leaveat_2 = tokenizer.decode(outputs[0][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
                                if lev(train_leaveat, train_leaveat_2) <= 5:
                                    verify_train = 1
                                    cont_train_certo = cont_train_certo + 1
                                    cont_train_leaveat_certo = cont_train_leaveat_certo + 1
                                elif train_leaveat == "":
                                    verify_train = 0
                                    cont_train_filledincorrect = cont_train_filledincorrect + 1
                                else:
                                    verify_train = 0
                                    cont_train_errado = cont_train_errado + 1
                                    cont_train_leaveat_errado = cont_train_leaveat_errado + 1
                            if i == 5:
                                if train_leaveat != "" and train_leaveat_2 == "" and tr_leaveat == 1:
                                    verify_train = 0
                                    tr_leaveat = 0
                                    cont_train_notfilled = cont_train_notfilled + 1
                        if verify_train == 1:
                            cont_turns_correct_train = cont_turns_correct_train + 1
                        if verify_train == 0:
                            cont_turns_incorrect_train = cont_turns_incorrect_train + 1
# Writing in the files the results obtained from the model for each individual slot as well as the results of the JGA and Slot F1 metrics
header = [model_name, 'Joint Goal Accuracy', 'F1', 'Precision', 'Recall']
with open('C:/Users/Francisco Pais/Desktop/multiwozPortugues-main/multiwozPortugues-main/Evaluation/Test/portuguese-t5-base/s_intent/results_lev_5.csv',
          'a', encoding='UTF8', newline='') as f:
    with open("C:/Users/Francisco Pais/Desktop/multiwozPortugues-main/multiwozPortugues-main/Evaluation/Test/portuguese-t5-base/s_intent/Ev_by_slots_LV/results_lev_5.txt", "w") as file:
        writer = csv.writer(f)
        writer.writerow(header)
        # ATTRACTION
        precision = round(cont_attraction_certo / (cont_attraction_certo + cont_attraction_filledincorrect), 2)
        recall = round(cont_attraction_certo / (cont_attraction_certo + cont_attraction_notfilled), 2)
        f1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        jga = round(cont_turns_correct_attraction / (cont_turns_correct_attraction + cont_turns_incorrect_attraction), 2)
        data = ["-", jga, f1, precision, recall]
        writer.writerow(data)
        print("Service Attraction Joint Goal Accuracy ", jga)
        print("Service Attraction Precision: ", precision)
        print("Service Attraction Recall: ", recall)
        print("Service Attraction F1: ", f1)
        print("Score by slots: ",
              "\n", "Slot Area: ", cont_attraction_area_certo / (cont_attraction_area_certo + cont_attraction_area_errado),
              "\n", "Slot Name: ", cont_attraction_name_certo / (cont_attraction_name_certo + cont_attraction_name_errado),
              "\n", "Slot Type: ", cont_attraction_type_certo / (cont_attraction_type_certo + cont_attraction_type_errado))

        file.write("Slot Area:"+" "+    str(cont_attraction_area_certo / (cont_attraction_area_certo + cont_attraction_area_errado))+ "\n")
        file.write("Slot Name:" + " " + str(cont_attraction_name_certo / (cont_attraction_name_certo + cont_attraction_name_errado)) + "\n")
        file.write("Slot Type:" + " " + str(cont_attraction_type_certo / (cont_attraction_type_certo + cont_attraction_type_errado))+ "\n")
        file.write("\n")
        # HOTEL
        print("\n")
        precision = round(cont_hotel_certo / (cont_hotel_certo + cont_hotel_filledincorrect), 2)
        recall = round(cont_hotel_certo / (cont_hotel_certo + cont_hotel_notfilled), 2)
        f1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        jga = round(cont_turns_correct_hotel / (cont_turns_correct_hotel + cont_turns_incorrect_hotel), 2)
        data = ["-", jga, f1, precision, recall]
        writer.writerow(data)
        print("Service Hotel Joint Goal Accuracy ", jga)
        print("Service Hotel Precision: ", precision)
        print("Service Hotel Recall: ", recall)
        print("Service Hotel F1: ", f1)
        print("Score by slots: ",
              "\n", "Slot Area: ",          cont_hotel_area_certo / (cont_hotel_area_certo + cont_hotel_area_errado),
              "\n", "Slot BookDay: ",      cont_hotel_bookday_certo / (cont_hotel_bookday_certo + cont_hotel_bookday_errado),
              "\n", "Slot BookPeople: ",   cont_hotel_bookpeople_certo / (cont_hotel_bookpeople_certo + cont_hotel_bookpeople_errado),
              "\n", "Slot BookStay: ",     cont_hotel_bookstay_certo / (cont_hotel_bookstay_certo + cont_hotel_bookstay_errado),
              "\n", "Slot Internet: ",      cont_hotel_internet_certo / (cont_hotel_internet_certo + cont_hotel_internet_errado),
              "\n", "Slot Name: ",          cont_hotel_name_certo / (cont_hotel_name_certo + cont_hotel_name_errado),
              "\n", "Slot Parking: ",       cont_hotel_parking_certo / (cont_hotel_parking_certo + cont_hotel_parking_errado),
              "\n", "Slot PriceRange: ",    cont_hotel_pricerange_certo / (cont_hotel_pricerange_certo + cont_hotel_pricerange_errado),
              "\n", "Slot Stars: ",         cont_hotel_stars_certo / (cont_hotel_stars_certo + cont_hotel_stars_errado),
              "\n", "Slot Type: ",          cont_hotel_type_certo / (cont_hotel_type_certo + cont_hotel_type_errado))

        file.write("Slot Area:" + " " +         str(cont_hotel_area_certo / (cont_hotel_area_certo + cont_hotel_area_errado)) + "\n")
        file.write("Slot BookDay:" + " " +     str(cont_hotel_bookday_certo / (cont_hotel_bookday_certo + cont_hotel_bookday_errado)) + "\n")
        file.write("Slot BookPeople:" + " " +  str(cont_hotel_bookpeople_certo / (cont_hotel_bookpeople_certo + cont_hotel_bookpeople_errado)) + "\n")
        file.write("Slot BookStay:" + " " +    str(cont_hotel_bookstay_certo / (cont_hotel_bookstay_certo + cont_hotel_bookstay_errado)) + "\n")
        file.write("Slot Internet:" + " " +     str(cont_hotel_internet_certo / (cont_hotel_internet_certo + cont_hotel_internet_errado)) + "\n")
        file.write("Slot Name:" + " " +         str(cont_hotel_name_certo / (cont_hotel_name_certo + cont_hotel_name_errado)) + "\n")
        file.write("Slot Parking:" + " " +      str(cont_hotel_parking_certo / (cont_hotel_parking_certo + cont_hotel_parking_errado))+ "\n")
        file.write("Slot PriceRange:" + " " +   str(cont_hotel_pricerange_certo / (cont_hotel_pricerange_certo + cont_hotel_pricerange_errado)) + "\n")
        file.write("Slot Stars:" + " " +        str(cont_hotel_stars_certo / (cont_hotel_stars_certo + cont_hotel_stars_errado)) + "\n")
        file.write("Slot Type:" + " " +         str(cont_hotel_type_certo / (cont_hotel_type_certo + cont_hotel_type_errado)) + "\n")
        file.write("\n")
        # RESTAURANT
        print("\n")
        precision = round(cont_restaurant_certo / (cont_restaurant_certo + cont_restaurant_filledincorrect), 2)
        recall = round(cont_restaurant_certo / (cont_restaurant_certo + cont_restaurant_notfilled), 2)
        f1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        jga = round(cont_turns_correct_restaurant / (cont_turns_correct_restaurant + cont_turns_incorrect_restaurant), 2)
        data = ["-", jga, f1, precision, recall]
        writer.writerow(data)
        print("Service Restaurant Joint Goal Accuracy ", jga)
        print("Service Restaurant Precision: ", precision)
        print("Service Restaurant Recall: ", recall)
        print("Service Restaurant F1: ", f1)
        print("Score by slots: ",
              "\n", "Slot Area: ",          cont_restaurant_area_certo / (cont_restaurant_area_certo + cont_restaurant_area_errado),
              "\n", "Slot BookDay: ",      cont_restaurant_bookday_certo / (cont_restaurant_bookday_certo + cont_restaurant_bookday_errado),
              "\n", "Slot BookPeople: ",   cont_restaurant_bookpeople_certo / (cont_restaurant_bookpeople_certo + cont_restaurant_bookpeople_errado),
              "\n", "Slot BookTime: ",     cont_restaurant_booktime_certo / (cont_restaurant_booktime_certo + cont_restaurant_booktime_errado),
              "\n", "Slot Food: ",          cont_restaurant_food_certo / (cont_restaurant_food_certo + cont_restaurant_food_errado),
              "\n", "Slot Name: ",          cont_restaurant_name_certo / (cont_restaurant_name_certo + cont_restaurant_name_errado),
              "\n", "Slot PriceRange: ",    cont_restaurant_pricerange_certo / (cont_restaurant_pricerange_certo + cont_restaurant_pricerange_errado))

        file.write("Slot Area:" + " " +         str(cont_restaurant_area_certo / (cont_restaurant_area_certo + cont_restaurant_area_errado)) + "\n")
        file.write("Slot BookDay:" + " " +     str(cont_restaurant_bookday_certo / (cont_restaurant_bookday_certo + cont_restaurant_bookday_errado)) + "\n")
        file.write("Slot BookPeople:" + " " +  str(cont_restaurant_bookpeople_certo / (cont_restaurant_bookpeople_certo + cont_restaurant_bookpeople_errado)) + "\n")
        file.write("Slot BookTime:" + " " +    str(cont_restaurant_booktime_certo / (cont_restaurant_booktime_certo + cont_restaurant_booktime_errado)) + "\n")
        file.write("Slot Food:" + " " +         str(cont_restaurant_food_certo / (cont_restaurant_food_certo + cont_restaurant_food_errado)) + "\n")
        file.write("Slot Name:" + " " +         str(cont_restaurant_name_certo / (cont_restaurant_name_certo + cont_restaurant_name_errado)) + "\n")
        file.write("Slot PriceRange:" + " " +   str(cont_restaurant_pricerange_certo / (cont_restaurant_pricerange_certo + cont_restaurant_pricerange_errado)) + "\n")
        file.write("\n")
        # TAXI
        print("\n")
        precision = round(cont_taxi_certo / (cont_taxi_certo + cont_taxi_filledincorrect), 2)
        recall = round(cont_taxi_certo / (cont_taxi_certo + cont_taxi_notfilled), 2)
        f1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        jga = round(cont_turns_correct_taxi / (cont_turns_correct_taxi + cont_turns_incorrect_taxi), 2)
        data = ["-", jga, f1, precision, recall]
        writer.writerow(data)
        print("Service Taxi Joint Goal Accuracy ", jga)
        print("Service Taxi Precision: ", precision)
        print("Service Taxi Recall: ", recall)
        print("Service Taxi F1: ", f1)
        print("Score by slots: ",
              "\n", "Slot ArriveBy: ",         cont_taxi_arriveby_certo / (cont_taxi_arriveby_certo + cont_taxi_arriveby_errado),
              "\n", "Slot Departure: ",         cont_taxi_departure_certo / (cont_taxi_departure_certo + cont_taxi_departure_errado),
              "\n", "Slot Destination: ",       cont_taxi_destination_certo / (cont_taxi_destination_certo + cont_taxi_destination_errado),
              "\n", "Slot LeaveAt: ",          cont_taxi_leaveat_certo / (cont_taxi_leaveat_certo + cont_taxi_leaveat_errado))

        file.write("Slot ArriveBy:" + " " +    str(cont_taxi_arriveby_certo / (cont_taxi_arriveby_certo + cont_taxi_arriveby_errado)) + "\n")
        file.write("Slot Departure:" + " " +    str(cont_taxi_departure_certo / (cont_taxi_departure_certo + cont_taxi_departure_errado)) + "\n")
        file.write("Slot Destination:" + " " +  str(cont_taxi_destination_certo / (cont_taxi_destination_certo + cont_taxi_destination_errado)) + "\n")
        file.write("Slot LeaveAt:" + " " +     str(cont_taxi_leaveat_certo / (cont_taxi_leaveat_certo + cont_taxi_leaveat_errado)) + "\n")
        file.write("\n")
        # TRAIN
        print("\n")
        precision = round(cont_train_certo / (cont_train_certo + cont_train_filledincorrect), 2)
        recall = round(cont_train_certo / (cont_train_certo + cont_train_notfilled), 2)
        f1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        jga = round(cont_turns_correct_train / (cont_turns_correct_train + cont_turns_incorrect_train), 2)
        data = ["-", jga, f1, precision, recall]
        writer.writerow(data)
        print("Service Train Joint Goal Accuracy ", jga)
        print("Service Train Precision: ", precision)
        print("Service Train Recall: ", recall)
        print("Service Train F1: ", f1)
        print("Score by slots: ",
              "\n", "Slot ArriveBy: ",     cont_train_arriveby_certo / (cont_train_arriveby_certo + cont_train_arriveby_errado),
              "\n", "Slot BookPeople: ",   cont_train_bookpeople_certo / (cont_train_bookpeople_certo + cont_train_bookpeople_errado),
              "\n", "Slot Day: ",           cont_train_day_certo / (cont_train_day_certo + cont_train_day_errado),
              "\n", "Slot Departure: ",     cont_train_departure_certo / (cont_train_departure_certo + cont_train_departure_errado),
              "\n", "Slot Destination: ",   cont_train_destination_certo / (cont_train_destination_certo + cont_train_destination_errado),
              "\n", "Slot LeaveAt: ",      cont_train_leaveat_certo / (cont_train_leaveat_certo + cont_train_leaveat_errado))

        file.write("Slot ArriveBy:" + " " +    str(cont_train_arriveby_certo / (cont_train_arriveby_certo + cont_train_arriveby_errado)) + "\n")
        file.write("Slot BookPeople:" + " " +  str(cont_train_bookpeople_certo / (cont_train_bookpeople_certo + cont_train_bookpeople_errado)) + "\n")
        file.write("Slot Day:" + " " +          str(cont_train_day_certo / (cont_train_day_certo + cont_train_day_errado)) + "\n")
        file.write("Slot Departure:" + " " +    str(cont_train_departure_certo / (cont_train_departure_certo + cont_train_departure_errado))+ "\n")
        file.write("Slot Destination:" + " " +  str(cont_train_destination_certo / (cont_train_destination_certo + cont_train_destination_errado)) + "\n")
        file.write("Slot LeaveAt:" + " " +     str(cont_train_leaveat_certo / (cont_train_leaveat_certo + cont_train_leaveat_errado)) + "\n")
        print(cont_turns_correct_attraction+cont_turns_incorrect_attraction+cont_turns_correct_hotel+cont_turns_incorrect_hotel+cont_turns_correct_restaurant+cont_turns_incorrect_restaurant+cont_turns_correct_taxi+cont_turns_incorrect_taxi+cont_turns_correct_train+cont_turns_incorrect_train)