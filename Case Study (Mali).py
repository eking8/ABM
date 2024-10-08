"""

 This script was written for the use of the GRASP case study in Mali. However, it can serve as
a template for instance construction. Follow this script and read all speeched comments. 
You will also need to alter Agents.py for your case:
- Import population data
- Import wealth distribution statistics
- Import gender distribution data (optional)
- Include camp closing logic (Fassala in the Mali example)

 For bug fixing, Network.py includes additional functions to probe diagnosis.

"""



import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx
import csv
import time
from Locations import City, Camp, total_pop
from Agents import Agent, deathmech
from Network import create_graph, draw_graph
from Visualisations import colors, print_progress_bar
from datetime import datetime, timedelta
import sys

""" 

Population Data

Create cities for the network with their Geocode name, population, and presence of an airport.
With Mali, the top 20 most populous cities were chosen alongside significant airports.

Do the same for foreign cities. Manually choose these by scanning along boarder in map.

"""

# populations from 2009 census https://www.citypopulation.de/en/mali/cities/

bamako = City("Bamako", hasairport=True, population = 1810366)
sikasso = City("Sikasso", hasairport=True, population = 226618)
koutiala = City("Koutiala", hasairport=True, population = 141444)
segou = City("Segou", population = 0.0321*133501)
kayes = City("Kayes", hasairport=True, population = 126319)
gao = City("Gao", hasairport=True, population = 86353)
san = City("San", population = 66967)
bougouni = City("Bougouni", population = 58538)
tombouctou = City("Timbuktu", hasairport=True, population = 54629)
kita = City("Kita", population = 49043)

niono_socoura= City("Niono", population = 34113)
mopti = City("Mopti", hasairport=True, population=120786)
koulikoro = City("Koulikoro", population=41602)
fana = City("Fana", population=36854)
nioro = City("Nioro", hasairport=True, population=33691)
kidal = City("Kidal", population=25969)
douentza = City("Douentza", hasairport=True, population=24005)
kadiolo = City("Kadiolo", population=24749)
djenne = City("Djenne", population=26267)
zegoua = City("Zegoua", population=20496)

# Create list of city classes
cities = [bamako, sikasso, koutiala, segou, kayes, gao, san, bougouni, tombouctou, kita,
          niono_socoura, mopti, koulikoro, fana, nioro, kidal, douentza, kadiolo, djenne, zegoua]


# 2013 census https://www.citypopulation.de/en/senegal/cities/

# Senegal
bakel = City("Bakel", country="Senegal", population=13329)
tambacounda = City("Tambacounda", country="Senegal", population=107293)
kedougou = City("Kedougou", country="Senegal", population=30051)

# 2014 Census https://www.citypopulation.de/en/guinea/cities/

# Guinea
dinguiraye = City("Dinguiraye", country="Guinea", population=18082)
siguiri = City("Siguiri", country="Guinea", population=127492)
fodekaria = City("Fodekaria", country="Guinea", population=20112)
kankan = City("Kankan", country="Guinea", population=190722)
mandiana = City("Mandiana", country="Guinea", population=16460)

# 2014 census https://www.citypopulation.de/en/ivorycoast/cities/

# Côte D'ivoire
odienne = City("Odienne", country="Côte D'ivoire", population=42173)
tingrela = City("Tingrela", country="Côte D'ivoire", population=40323)
boundiali = City("Boundiali", country="Côte D'ivoire", population=39962)
korhogo = City("Korhogo", country="Côte D'ivoire", population=243048)

# Average of 2006 and 2019

# Burkina Faso
banfora = City("Banfora", country="Burkina Faso", population=(75917+117452)/2)
dande = City("Dandé", country="Burkina Faso", population=(0)/2)
solenzo = City("Solenzo", country="Burkina Faso", population=(16850+24783)/2)
nouna = City("Nouna", country="Burkina Faso", population=(22166+32428)/2)
dedougou = City("Dedougou", country="Burkina Faso", population=(38862+63617)/2)
tougan = City("Tougan", country="Burkina Faso", population=(17050+26347)/2)
ouahigouya = City("Ouahigouya", country="Burkina Faso", population=(73153+124587)/2)
arbinda = City("Arbinda", country="Burkina Faso", population=(9790	+31215)/2)
dori = City("Dori", country="Burkina Faso", population=(21078+46512)/2)

# 2012 census https://www.citypopulation.de/en/niger/cities/
# Niger
ayorou = City("Ayorou", country="Niger", population=11528)
tera = City("Tera", country="Niger", population=29119)
filingue = City("Filingué", country="Niger", population=12224)
assamakka = City("Assamakka", country="Niger", population=0)

# 2008 census https://www.citypopulation.de/en/algeria/cities/

# Algeria
bordj_badji_mokhtar = City("Bordj Badji Mokhtar", country="Algeria", population=628475)

# 2013 census https://www.citypopulation.de/en/mauritania/cities/

# Mauritania
walatah = City("Walatah", country="Mauritania", population=0)
nema = City("Néma", country="Mauritania", population=21708)
timbedra = City("Timbédra", country="Mauritania", population=14131)
tchera_rouissa = City("Tchera Rouissa", country="Mauritania", population=0)
adel_bagrou = City("Adel Bagrou", country="Mauritania", population=8576)
tintane = City("Tintane", country="Mauritania", population=12690)
selibaby = City("Sélibaby", country="Mauritania", population=26420)




foreign_cities = [
    bakel, tambacounda, kedougou,
    dinguiraye, siguiri, fodekaria, kankan, mandiana,
    odienne, tingrela, boundiali, korhogo,
    banfora, dande, solenzo, nouna, dedougou, tougan, ouahigouya, arbinda, dori,
    ayorou, tera, filingue, assamakka,
    bordj_badji_mokhtar,
    walatah, nema, timbedra, tchera_rouissa, adel_bagrou, tintane, selibaby
]


""" 

Initialize camps wih camo name, country name, and population

"""

# Define camps
bobo = Camp("Bobo", "Burkina Faso", population=0) # 2012-02-29
goudoubo = Camp("Goudoubo", "Burkina Faso",population=0) # 2012-02-29
mentao = Camp("Mentao", "Burkina Faso", population=0) # 2012-02-29
ouagadougou = Camp("Ouagadougou", "Burkina Faso", population=0) # 2012-02-29
fassala = Camp("Fassala", "Mauritania", population= 20000) # 2012-02-29
mbera = Camp("Mbera", "Mauritania", population=0) # 2012-02-29
abala = Camp("Abala", "Niger", population=1881) # 2012-02-29
intikane = Camp("Intikane", "Niger", population=5058) # 2013-05-07
mangaize = Camp("Mangaize", "Niger", population=1140) # 2012-02-29
niamey = Camp("Niamey", "Niger", population=0) # 2012-02-29
tabareybarey = Camp("Tabareybarey", "Niger", population=0) # 2012-02-29

camps = [bobo, goudoubo, mentao, ouagadougou, fassala, mbera, abala, intikane, mangaize, niamey, tabareybarey]

# populations from 2009 census https://www.citypopulation.de/en/mali/cities/

""" 

Additional airports not considered before (outside top-20 most populous cities)

"""

# Define airports
kenieba = City("Kéniéba", hasairport=True, top20=False, population=0)
yelimane = City("Yélimané", hasairport=True, top20=False, population=0)
ansongo = City("Ansongo", hasairport=True, top20=False, population=0)
bafoulabe = City("Bafoulabé", hasairport=True, top20=False, population=0)
goundam = City("Goundam", hasairport=True, top20=False, population=12586)
tessalit = City("Tessalit", hasairport=True, top20=False, population=0)
bourem = City("Bourem", hasairport=True, top20=False, population=27488)
bandiagara = City("Bandiagara", hasairport=True, top20=False, population=17166)
bengassi = City("Bengassi", hasairport=True, top20=False, population=0)
menaka = City("Menaka", hasairport=True, top20=False, population=9138)

airports = [mopti, kenieba, yelimane, menaka, douentza, ansongo, bafoulabe, goundam, tessalit, bourem, bandiagara, bengassi, nioro]


"""

Further cities added for granularity

"""

# populations from 2009 census https://www.citypopulation.de/en/mali/cities/ 

# Non-top 20 cities

lere = City("Lere", top20=False, population=0)
syama = City("Syama", top20=False, population=0)
diabaly = City("Diabaly", top20=False, population=0)
sevare = City("Sevare", top20=False, population=0)
nara = City("Nara", top20=False, population=15310)
niafunke = City("Niafunke", top20=False, population=0)
aguelhok = City("Aguelhok", top20=False, population=0)
koue = City("Koue", top20=False, population=0)
sari = City("Sari", top20=False, population=0)
diago = City("Diago", top20=False, population=0)
ber = City("Ber", top20=False, population=0)
anefis = City("Anefis", top20=False, population=0)
dire = City("Dire", top20=False, population=20337)
tenenkou = City("Tenenkou", top20=False, population=0)
youwarou = City("Youwarou", top20=False, population=0)
hombori = City("Hombori", top20=False, population=0)
tin_zaouaten = City("Tinzaouaten", top20=False, population=0)
anderamboukane = City("Anderamboukane", top20=False, population=0)


cities.append(lere)
cities.append(syama)
cities.append(diabaly)
cities.append(sevare)
cities.append(nara)
cities.append(niafunke)
cities.append(aguelhok)
cities.append(koue)
cities.append(sari)
cities.append(diago)
cities.append(ber)
cities.append(anefis)
cities.append(dire)
cities.append(tenenkou)
cities.append(youwarou)
cities.append(hombori)
cities.append(tin_zaouaten)
cities.append(anderamboukane)

cities += airports
# Combine all locations into a single list
locations = cities + camps + foreign_cities


"""

 At this stage use the draw graph function to display your network. Depending on your chosen level of 
granularity, go through manually each node and find the nodes you would like to build a connection
with. A useful tip would be to consider an arbritrary maximum distance that an agent can travel 
between nodes (bear in mind agents can travel multiple nodes).

"""

kayes.add_connection(kita)
kayes.add_connection(segou)
kayes.add_connection(bamako)
kita.add_connection(segou)
kita.add_connection(bamako)
kita.add_connection(san)
bamako.add_connection(segou)
bamako.add_connection(san)
bamako.add_connection(koutiala)
bamako.add_connection(bougouni)
bamako.add_connection(sikasso)
bougouni.add_connection(sikasso)
sikasso.add_connection(koutiala)
segou.add_connection(san)
segou.add_connection(koutiala)
segou.add_connection(tombouctou)
segou.add_connection(gao)
koutiala.add_connection(san)
koutiala.add_connection(gao)
san.add_connection(gao)
gao.add_connection(tombouctou)
korhogo.add_connection(zegoua)
korhogo.add_connection(banfora)
korhogo.add_connection(tingrela)
boundiali.add_connection(tingrela)
nioro.add_connection(kayes)
nioro.add_connection(mbera)
nioro.add_connection(fassala)
nioro.add_connection(kita)
nioro.add_connection(koulikoro)
nioro.add_connection(segou)
nioro.add_connection(niono_socoura)
koulikoro.add_connection(kayes)
koulikoro.add_connection(kita)
koulikoro.add_connection(bamako)
koulikoro.add_connection(koutiala)
koulikoro.add_connection(san)
koulikoro.add_connection(segou)
koulikoro.add_connection(niono_socoura)
niono_socoura.add_connection(kayes)
niono_socoura.add_connection(mbera)
niono_socoura.add_connection(fassala)
niono_socoura.add_connection(tombouctou)
niono_socoura.add_connection(fana)
niono_socoura.add_connection(douentza)
niono_socoura.add_connection(mopti)
niono_socoura.add_connection(djenne)
mopti.add_connection(djenne)
mopti.add_connection(segou)
mopti.add_connection(fassala)
mopti.add_connection(fana)
mopti.add_connection(douentza)
mopti.add_connection(goudoubo)
mopti.add_connection(mentao)
mopti.add_connection(ouagadougou)
mopti.add_connection(bobo)
fana.add_connection(fassala)
fana.add_connection(mbera)
fana.add_connection(tombouctou)
fana.add_connection(gao)
fana.add_connection(douentza)
fana.add_connection(mopti)
fana.add_connection(djenne)
fana.add_connection(bamako)
fana.add_connection(koulikoro)
fana.add_connection(sikasso)
fana.add_connection(koutiala)
fana.add_connection(san)
kidal.add_connection(tombouctou)
kidal.add_connection(gao)
kidal.add_connection(intikane)
douentza.add_connection(fassala)
douentza.add_connection(mbera)
douentza.add_connection(fana)
douentza.add_connection(gao)
douentza.add_connection(tabareybarey)
douentza.add_connection(goudoubo)
douentza.add_connection(mentao)
kadiolo.add_connection(zegoua)
kadiolo.add_connection(bougouni)
kadiolo.add_connection(sikasso)
kadiolo.add_connection(koutiala)
kadiolo.add_connection(bobo)
zegoua.add_connection(bobo)
zegoua.add_connection(sikasso)
zegoua.add_connection(koutiala)
sari.add_connection(mopti)
sari.add_connection(hombori)
sari.add_connection(ouahigouya)
sari.add_connection(mentao)
sari.add_connection(arbinda)
sari.add_connection(douentza)
mbera.add_connection(youwarou)
mbera.add_connection(dire)
fassala.add_connection(timbedra)
fassala.add_connection(sevare)
abala.add_connection(ansongo)
abala.add_connection(ayorou)
abala.add_connection(tera)
mangaize.add_connection(ansongo)
mangaize.add_connection(menaka)
mangaize.add_connection(dori)
niamey.add_connection(anderamboukane)
ouagadougou.add_connection(arbinda)
ouagadougou.add_connection(dori)
ouagadougou.add_connection(lere)
mentao.add_connection(lere)
mentao.add_connection(koue)
mentao.add_connection(dedougou)
bobo.add_connection(tingrela)
bobo.add_connection(syama)
bobo.add_connection(fana)
syama.add_connection(sikasso)
tingrela.add_connection(sikasso)
hombori.add_connection(dire)
hombori.add_connection(douentza)
hombori.add_connection(gao)
hombori.add_connection(ansongo)
hombori.add_connection(tabareybarey)
hombori.add_connection(ayorou)
hombori.add_connection(goudoubo)
hombori.add_connection(arbinda)
hombori.add_connection(mentao)
ansongo.add_connection(gao)
anderamboukane.add_connection(ansongo)
anderamboukane.add_connection(menaka)
anderamboukane.add_connection(intikane)
anderamboukane.add_connection(abala)
anderamboukane.add_connection(mangaize)
ouahigouya.add_connection(mentao)
ouahigouya.add_connection(ouagadougou)
ouahigouya.add_connection(tougan)
ouahigouya.add_connection(lere)
lere.add_connection(tougan)
lere.add_connection(san)
sari.add_connection(bandiagara)
arbinda.add_connection(bandiagara)
diabaly.add_connection(mbera)
nioro.add_connection(timbedra)
nema.add_connection(goundam)
nema.add_connection(niafunke)
tin_zaouaten.add_connection(menaka)
tin_zaouaten.add_connection(aguelhok)
tin_zaouaten.add_connection(anefis)
menaka.add_connection(assamakka)
ansongo.add_connection(menaka)
bourem.add_connection(menaka)
hombori.add_connection(niafunke)
ber.add_connection(tombouctou)
ber.add_connection(bourem)
ber.add_connection(gao)
ber.add_connection(anefis)
ber.add_connection(kidal)
ber.add_connection(aguelhok)
bourem.add_connection(menaka)
tera.add_connection(ouagadougou)
dori.add_connection(mentao)
anderamboukane.add_connection(tabareybarey)
arbinda.add_connection(tabareybarey)
youwarou.add_connection(dire)
youwarou.add_connection(douentza)
sevare.add_connection(douentza)
douentza.add_connection(dire)
douentza.add_connection(youwarou)
douentza.add_connection(sevare)
dire.add_connection(goundam)
dire.add_connection(tombouctou)
dire.add_connection(ber)
sevare.add_connection(djenne)
sevare.add_connection(koue)
koue.add_connection(nouna)
djenne.add_connection(san)
lere.add_connection(nouna)
tougan.add_connection(dedougou)
dedougou.add_connection(bobo)
dande.add_connection(banfora)
diago.add_connection(kita)
diago.add_connection(koulikoro)
diago.add_connection(san)
sari.add_connection(tabareybarey)
sari.add_connection(ayorou)
ouahigouya.add_connection(niamey)
tera.add_connection(filingue)
tera.add_connection(mangaize)
youwarou.add_connection(hombori)
douentza.add_connection(ayorou)
hombori.add_connection(anderamboukane)
kayes.add_connection(nara)
nara.add_connection(tchera_rouissa)
nara.add_connection(adel_bagrou)
nara.add_connection(mbera)
nara.add_connection(fassala)
dire.add_connection(goundam)
dire.add_connection(niafunke)
niafunke.add_connection(goundam)
niafunke.add_connection(youwarou)
niafunke.add_connection(mbera)
niafunke.add_connection(fassala)
youwarou.add_connection(fassala)
youwarou.add_connection(diabaly)
youwarou.add_connection(tenenkou)
youwarou.add_connection(sevare)
youwarou.add_connection(mopti)
diabaly.add_connection(nara)
diabaly.add_connection(niono_socoura)
diabaly.add_connection(fassala)
diabaly.add_connection(tenenkou)
tenenkou.add_connection(djenne)
tenenkou.add_connection(niono_socoura)
tenenkou.add_connection(fassala)
tenenkou.add_connection(sevare)
tenenkou.add_connection(bandiagara)
sevare.add_connection(mopti)
sevare.add_connection(bandiagara)
koue.add_connection(san)
koue.add_connection(djenne)
koue.add_connection(lere)
koue.add_connection(bandiagara)
lere.add_connection(san)
lere.add_connection(bandiagara)
diago.add_connection(fana)
diago.add_connection(bamako)
diago.add_connection(koutiala)
bengassi.add_connection(kenieba)
bengassi.add_connection(tambacounda)
bengassi.add_connection(bafoulabe)
bengassi.add_connection(kayes)
kenieba.add_connection(kedougou)
bafoulabe.add_connection(tambacounda)
mbera.add_connection(tombouctou)
mbera.add_connection(fassala)
fassala.add_connection(kayes)
fassala.add_connection(tombouctou)
bobo.add_connection(bougouni)
bobo.add_connection(sikasso)
bobo.add_connection(koutiala)
bobo.add_connection(ouagadougou)
ouagadougou.add_connection(sikasso)
ouagadougou.add_connection(koutiala)
ouagadougou.add_connection(san)
ouagadougou.add_connection(mentao)
ouagadougou.add_connection(goudoubo)
ouagadougou.add_connection(niamey)
niamey.add_connection(san)
niamey.add_connection(mentao)
niamey.add_connection(goudoubo)
niamey.add_connection(tabareybarey)
niamey.add_connection(mangaize)
niamey.add_connection(abala)
mentao.add_connection(segou)
mentao.add_connection(san)
mentao.add_connection(goudoubo)
goudoubo.add_connection(tabareybarey)
goudoubo.add_connection(mangaize)
mangaize.add_connection(tabareybarey)
mangaize.add_connection(abala)
tabareybarey.add_connection(abala)
tabareybarey.add_connection(intikane)
tabareybarey.add_connection(gao)
abala.add_connection(gao)
abala.add_connection(intikane)
intikane.add_connection(gao)
yelimane.add_connection(nioro)
yelimane.add_connection(kayes)
yelimane.add_connection(koulikoro)
bafoulabe.add_connection(kayes)
bafoulabe.add_connection(kita)
bengassi.add_connection(kita)
kenieba.add_connection(kita)
kenieba.add_connection(bamako)
goundam.add_connection(mbera)
goundam.add_connection(fassala)
goundam.add_connection(tombouctou)
goundam.add_connection(douentza)
bourem.add_connection(tombouctou)
bourem.add_connection(gao)
bourem.add_connection(kidal)
menaka.add_connection(kidal)
menaka.add_connection(gao)
menaka.add_connection(intikane)
ansongo.add_connection(tabareybarey)
menaka.add_connection(gao)
menaka.add_connection(douentza)
bandiagara.add_connection(mopti)
bandiagara.add_connection(segou)
bandiagara.add_connection(djenne)
bandiagara.add_connection(mentao)
bandiagara.add_connection(niono_socoura)
kenieba.add_connection(bengassi)
kenieba.add_connection(kedougou)
kenieba.add_connection(tambacounda)
kenieba.add_connection(dinguiraye)
bengassi.add_connection(tambacounda)
bengassi.add_connection(bafoulabe)
selibaby.add_connection(yelimane)
selibaby.add_connection(nioro)
selibaby.add_connection(tintane)
selibaby.add_connection(bakel)
bakel.add_connection(kayes)
bakel.add_connection(bafoulabe)
bakel.add_connection(tambacounda)
tambacounda.add_connection(yelimane)
tambacounda.add_connection(kayes)
kedougou.add_connection(kayes)
kedougou.add_connection(kita)
kedougou.add_connection(bamako)
kedougou.add_connection(dinguiraye)
kedougou.add_connection(tambacounda)
dinguiraye.add_connection(bamako)
dinguiraye.add_connection(kita)
dinguiraye.add_connection(siguiri)
dinguiraye.add_connection(fodekaria)
dinguiraye.add_connection(kankan)
siguiri.add_connection(bamako)
siguiri.add_connection(bougouni)
siguiri.add_connection(fodekaria)
fodekaria.add_connection(bougouni)
fodekaria.add_connection(kankan)
fodekaria.add_connection(mandiana)
kankan.add_connection(mandiana)
mandiana.add_connection(odienne)
mandiana.add_connection(tingrela)
mandiana.add_connection(boundiali)
mandiana.add_connection(bougouni)
odienne.add_connection(tingrela)
odienne.add_connection(boundiali)
tingrela.add_connection(bougouni)
tingrela.add_connection(syama)
tingrela.add_connection(kadiolo)
tingrela.add_connection(zegoua)
syama.add_connection(kadiolo)
syama.add_connection(bougouni)
syama.add_connection(banfora)
odienne.add_connection(sikasso)
odienne.add_connection(kadiolo)
odienne.add_connection(zegoua)
odienne.add_connection(boundiali)
odienne.add_connection(korhogo)
boundiali.add_connection(korhogo)
boundiali.add_connection(zegoua)
boundiali.add_connection(banfora)
banfora.add_connection(zegoua)
banfora.add_connection(kadiolo)
banfora.add_connection(bobo)
dande.add_connection(bobo)
dande.add_connection(sikasso)
dande.add_connection(koutiala)
dande.add_connection(solenzo)
dande.add_connection(ouagadougou)
solenzo.add_connection(bobo)
solenzo.add_connection(koutiala)
solenzo.add_connection(nouna)
solenzo.add_connection(dedougou)
solenzo.add_connection(ouagadougou)
dedougou.add_connection(koutiala)
dedougou.add_connection(nouna)
dedougou.add_connection(ouagadougou)
nouna.add_connection(koutiala)
nouna.add_connection(san)
nouna.add_connection(tougan)
nouna.add_connection(ouagadougou)
tougan.add_connection(san)
tougan.add_connection(djenne)
tougan.add_connection(ouagadougou)
tougan.add_connection(mentao)
arbinda.add_connection(mentao)
arbinda.add_connection(dori)
arbinda.add_connection(goudoubo)
dori.add_connection(goudoubo)
dori.add_connection(tera)
dori.add_connection(niamey)
tera.add_connection(goudoubo)
tera.add_connection(ayorou)
tera.add_connection(niamey)
filingue.add_connection(niamey)
filingue.add_connection(mangaize)
filingue.add_connection(abala)
filingue.add_connection(goudoubo)
ayorou.add_connection(tabareybarey)
ayorou.add_connection(goudoubo)
ayorou.add_connection(mangaize)
ayorou.add_connection(niamey)
assamakka.add_connection(intikane)
assamakka.add_connection(gao)
assamakka.add_connection(kidal)
assamakka.add_connection(bordj_badji_mokhtar)
bordj_badji_mokhtar.add_connection(kidal)
bordj_badji_mokhtar.add_connection(tessalit)
bordj_badji_mokhtar.add_connection(tin_zaouaten)
tessalit.add_connection(kidal)
tessalit.add_connection(tin_zaouaten)
tin_zaouaten.add_connection(tessalit)
tin_zaouaten.add_connection(kidal)
tin_zaouaten.add_connection(assamakka)
aguelhok.add_connection(kidal)
aguelhok.add_connection(anefis)
anefis.add_connection(bourem)
anefis.add_connection(gao)
anefis.add_connection(menaka)
tintane.add_connection(tchera_rouissa)
tintane.add_connection(yelimane)
tintane.add_connection(nioro)
tintane.add_connection(timbedra)
tintane.add_connection(nema)
tintane.add_connection(walatah)
tchera_rouissa.add_connection(nioro)
tchera_rouissa.add_connection(timbedra)
tchera_rouissa.add_connection(adel_bagrou)
tchera_rouissa.add_connection(mbera)
adel_bagrou.add_connection(timbedra)
adel_bagrou.add_connection(nema)
adel_bagrou.add_connection(mbera)
adel_bagrou.add_connection(fassala)
timbedra.add_connection(nema)
timbedra.add_connection(mbera)
nema.add_connection(walatah)
nema.add_connection(mbera)
nema.add_connection(tombouctou)
walatah.add_connection(mbera)
nema.add_connection(tombouctou)

"""

No need to change this. This is the birthing and grouping logic inherent in GRASPs architecture.

"""

def try_to_create_agent(i, birth_rate, loc_dic, Agents):
    for attempt in range(10000):
        rand_loc = Agent.randomloc()
        for id in loc_dic[rand_loc].members:
            member = Agents[id]
            if member.is_leader and 16 < member.age < 65 and member.gender == 'F':
                new_agent = Agent(i, age=0, is_leader=False, location=rand_loc)
                Agents[i] = new_agent
                member.fam.append(new_agent)
                member.group.append(new_agent)
                new_agent.status=member.status

                if not member.in_family:
                    member.in_family = True
                else:
                    for agent in member.fam:
                        agent.fam = member.fam

                if not member.ingroup:
                    member.ingroup = True
                else:
                    for agent in member.group:
                        agent.group = member.group

                return True  # Successfully created an agent
    return False  # Failed to create an agent after all attempts

def update_location_and_groups(i, loc_dic, Agents, total_agents, ags):
    if i in Agents:
        Agents[i].in_family = True
        Agents[i].in_group = True
        total_agents += 1
        loc_dic[Agents[i].location].addmember(i)
        ags.append(Agents[i])

"""

Adjust time frame based on your case instance.

"""
# Calculate the number of days in 2012
start_date = pd.to_datetime('2012-01-01')
end_date = pd.to_datetime('2013-01-01')
dates = pd.date_range(start_date, end_date)

"""

 Here is where you input ACLED data for your event series.
df1 - Conflict in the country
df2 - Conflict in neighbouring country (can be pre-processed to remove irrelevent data,
      in this case we removed irrelevant nodes and filtered to timeframe before importing)

"""

df1 = pd.read_csv("1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")[
    pd.read_csv("1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")['year']==2012]
df1['event_date'] = pd.to_datetime(df1['event_date'], format="%d %B %Y")


df2 = pd.read_csv("for_con.csv") # Foreign conflicts included in foreign nodes

df2['event_date'] = pd.to_datetime(df2['event_date'], format="%d-%b-%y")

conflicts = pd.concat([df1, df2], ignore_index=True)

conflicts.sort_values(by='event_date', inplace=True)

"""

 Please take care over this next section as it is largely bug-prone. loc_dic is initated to 
avoid discrepencies between the names of the nodes in our graph (derived from OSMNX) and 
ACLED. Also, if a city/camp is not in the OSNMX database, choose the closest available point
and use this dictionary to link the conflict name back to this node. This dictionary also
provides the status of abroad and death to be a None location.

"""

# Location dictionary required as ACLED names are not the same as OSMNX

loc_dic = {'Bamako':bamako, 
           'Kidal':kidal, 
           'Timbuktu':tombouctou, 
           'Gao':gao, 
           'Lere':lere, 
           'Menaka':menaka,
           'Syama Gold Mine':syama, 
           'Mopti':mopti,
           'Diabaly':diabaly, 
           'Koutiala':koutiala, 
           'Douentza':douentza,
           'Bourem':bourem, 
           'Sevare':sevare, 
           'Bougouni':bougouni, 
           'Ansongo':ansongo, 
           'Nara':nara, 
           'Niafunke':niafunke,
           'Aguelhok':aguelhok, 
           'Komeye Koukou':koue, 
           'Goundam':goundam, 
           'Doubabougou':bamako, 
           'Niono':niono_socoura,
           'Sari':sari, 
           'Kati':bamako, 
           'Kambila': bamako, 
           'Diago':diago, 
           'Ber':ber, 
           'Anefis':anefis, 
           'Dire':dire,
           'Tessalit':tessalit, 
           'Tenenkou':tenenkou, 
           'Imezzehene':kidal, 
           'Youwarou': youwarou, 
           'Hombori': hombori,
           'Tin Zaouaten': tin_zaouaten, 
           'Anderamboukane': anderamboukane, 
           'Tin Kaloumane':kidal,
           'Kayes':kayes,
           'Fana':fana,
           'Koulikoro':koulikoro,
           'Yélimané':yelimane,
           'San':san,
           'Kita':kita,
           'Sikasso':sikasso,
           'Djenne':djenne,
           'Segou':segou,
           'Nouna':nouna,
           'Mentao':mentao,
           'Dandé':dande,
           'Mbera':mbera,
           'Bobo':bobo,
           'Syama':syama,
           'Tougan':tougan,
           'Ouagadougou':ouagadougou,
           'Sélibaby':selibaby,
           'Bandiagara':bandiagara,
           'Tingrela':tingrela,
           'Banfora':banfora,
           'Dedougou':dedougou,
           'Fassala':fassala,
           'Solenzo':solenzo,
           'Arbinda':arbinda,
           'Niamey':niamey,
           'Nioro':nioro,
           'Goudoubo':goudoubo,
           'Koue':koue,
           'Tabareybarey':tabareybarey,
           'Tera':tera,
           'Mangaize':mangaize,
           'Abala':abala,
           'Ouahigouya':ouahigouya,
           'Dori':dori,
           'Ayorou':ayorou,
           'Kadiolo':kadiolo,
           'Zegoua':zegoua,
           'Intikane':intikane,
           'Filingué':filingue,
           'Bafoulabé':bafoulabe,
           'Bengassi':bengassi,
           'Kéniéba':kenieba,
           'Abroad':None,
           'Dead':None,
           'Kedougou':kedougou,
           'Siguiri':siguiri,
           'Dinguiraye':dinguiraye,
           'Mandiana':mandiana,
           "Fodekaria":fodekaria,
           "Néma":nema,
           "Bakel":bakel,
           "Tambacounda":tambacounda,
           "Adel Bagrou":adel_bagrou,
           "Timbédra":timbedra,
           "Bordj Badji Mokhtar":bordj_badji_mokhtar,
           "Tchera Rouissa":tchera_rouissa,
           "Odienne":odienne,
           'Boundiali':boundiali,
           "Tintane":tintane,
           "Korhogo":korhogo,
           "Assamakka":assamakka,
           "Tinzaouaten":tin_zaouaten,
           "Pobe Mengao":mentao,
           "Bobo-Dioulasso":bobo,
           "Tengandogo":goudoubo,
           "Passakongo":dedougou}

print("Initilising graph... \n")

start_time = time.time()

total_population = total_pop(cities)+28079

"""

 frac decides the number of agents, hence length of run-time. Please see supplementary notes
on the benefits of lower and higher frac values.

"""

#########################################################################################
frac = 300 # TO VARY
#########################################################################################

"""

The next section initialises everything necessary for the simulation.

"""

n_agents = int(total_population/frac)

for loc in locations:
    loc.population= int(loc.population/frac)

city_probabilities = {city.name: city.population / n_agents for city in cities}

prob_values = list(city_probabilities.values())

# Normalize the probabilities
normalized_prob_values = [float(i)/sum(prob_values) for i in prob_values]

if not np.isclose(sum(normalized_prob_values), 1.0):
    raise ValueError("Normalized probabilities do not sum closely to 1.")
 
G = create_graph(locations)
ongoing_conflicts = []

Agent.calculate_distributions()
Agent.initialise_cities(foreign_cities+camps)
Agent.initialise_populations(cities+camps,total_population)

Agents = {}
ags = []

for loc in cities+camps:
    loc.population=0

print("Creating Agents... \n")

agent_t1=time.time()
for i in list(range(1,n_agents+1)):
    Agents[i] = Agent(i)
    loc_dic[Agents[i].location].addmember(i)
    ags.append(Agents[i])
    print(f'\rProgress: {i} Agents', end='', flush=True)

agent_t2=time.time()
print("\n \n")
print("...finished in: "+ str(agent_t2-agent_t1) + "\n")


print("Forming families... \n")
o=0
fam_t1=time.time()
for agent in ags:
    agent.form_families(ags)
    o+=1
    print(f'\rProgress: {o} Agents', end='', flush=True)
print("\n \n")
fam_t2=time.time()

print("...finished in: "+ str(fam_t2-fam_t1) + "\n")
print("Joining dependents to families... \n")
o=0
depen_t1=time.time()
for agent in ags:
    agent.join_dependents(ags)
    o+=1
    print(f'\rProgress: {o} Agents', end='', flush=True)

print("\n \n")
depen_t2=time.time()

print("...finished in: "+ str(depen_t2-depen_t1) + "\n")

print("Forming family groups... \n")
o=0
fam_groups_t1=time.time()
for agent in ags: # must initialise new for loop to ensure all families initialised
    agent.form_fam_group()
    o+=1
    print(f'\rProgress: {o} Agents', end='', flush=True)
fam_groups_t2=time.time()
print("\n \n")
print("...finished in: "+ str(fam_groups_t2-fam_groups_t1) + "\n")

print("Forming strategic groups... \n")
o=0
strat_groups_t1=time.time()
for loc in locations:
    loc.form_strat_group(Agents)
    o+=1
    print(f'\rProgress: {o} Locations', end='', flush=True)
strat_groups_t2=time.time()
print("\n \n")
print("...finished in: "+ str(strat_groups_t2-strat_groups_t1) + "\n")

print("Finish defining groups... \n")


def_groups_t1=time.time()

for agent in ags: # must initialise new for loop to ensure all proabilities of travelling with fam are initialised
    agent.define_groups(ags)

def_groups_t2=time.time()

print("...finished in: "+ str(def_groups_t2-def_groups_t1) + "\n")

print("Speed and captial normalisation... \n")

speeds_t1=time.time()
for agent in ags: # must initialise new group to ensure all family travelling groups initialised
    agent.group_speeds_and_cap()
speeds_t2=time.time()

print("...finished in: "+ str(speeds_t2-speeds_t1) + "\n")

if (45.92*n_agents)>(365*1000):
    birth_rate = round((45.92*n_agents)/(365*1000))
    bpd = True
else:
    birth_rate = round(1/((45.92*n_agents)/(365*1000))) # derived from https://www.statista.com/statistics/977023/crude-birth-rate-in-mali/
    bpd = False

populations = {camp.name: [] for camp in camps}
countries = {x:[] for x in ['Burkina Faso','Côte D\'Ivoire','Guinea','Mauritania','Niger','Senegal','Mali','Other','Dead']}
statuses = {x:[] for x in ['Dead','IDP','Returnee','Refugee','Resident','Fleeing from conflict']}

total_agents = len(Agents)

days = 0

fin_sim_t=time.time()




print("Total set up finished in: "+ str(fin_sim_t-start_time) + "\n")


"""

We now have everything we need... the next section is the iterative logic day by day, agent by agent. 
Please remove all Fassala based logic (lines 1036-1048) and replace with any camp closure logic
manually. 1090-1136 must be changed to attribute your country of choice and boardering countries.

"""


print("Starting simulation...")

for current_date in dates: 

    processed_agents = 0
        
    print("\n")
    print(f"Simulating day: {current_date.strftime('%Y-%m-%d')}")
    
    if current_date==datetime(2012, 3, 19).date():
        for id in Agents:
            fassala.in_city_conflict(1000,datetime(2012, 3, 19).date())
        fassala.is_open=False
        fassala.iscamp=False
        nx.set_node_attributes(G, { 'Fassala': False }, 'is_open')
        nx.set_node_attributes(G, { 'Fassala': 'Closed camp' }, 'type')
        for id in fassala.members:
            Agents[id].moving = True
            Agents[id].status = 'Refugee'
            Agents[id].startdate=current_date
            Agents[id].longterm='Mbera'
            Agents[id].distance_traveled_since_rest=0
        

    for ongoing_conflict in ongoing_conflicts:
        if ongoing_conflict=='Fassala' and current_date>=datetime(2012, 3, 19).date():
            pass
        else:
            ongoing_conflict.check_and_update_conflict_status(current_date) # check ongoing conflicts
        if ongoing_conflict.hasconflict:
            pass
        else:
            ongoing_conflicts.remove(ongoing_conflict) # remove conflict from stored list
            G.nodes[ongoing_conflict.name]['has_conflict']=False # update graph node
            if ongoing_conflict.name in camps:
                for id in Agents:
                    if ongoing_conflict.name in Agents[id].nogos:
                        Agents[id].nogos.remove(ongoing_conflict.name)

    
    
    for idx, event in conflicts[conflicts['event_date'] == current_date].iterrows(): # consider all conflicts in given day

        location = loc_dic[event['location']]

        fat = event['fatalities']
        
        G.nodes[location.name]['has_conflict']=True

        G.nodes[location.name]['population']-= fat
        G.nodes[location.name]['fatalities']+= fat
        death_ids = deathmech(loc_dic[location.name].members,fat) 
        loc_dic[location.name].population -= fat
            
        if death_ids:
            for id in death_ids:
                Agents[id].kill(frac,ags)
        
        location.in_city_conflict(event['fatalities'], current_date)

        if location not in ongoing_conflicts:
            ongoing_conflicts.append(location)
    
    refugees=0
    returnees=0
    idps=0
    deaths=0
    residents=0
    fleeing=0

    Burkinas=0
    Cotes = 0
    Guineas = 0
    Maus = 0
    Nigers = 0
    Senegals = 0
    Malis = 0
    Others = 0

    for id in Agents:
                   
        Agents[id].assess_situation_and_move_if_needed(G,loc_dic[Agents[id].location],current_date,roulette=True)
                
        if Agents[id].status != 'Dead' and Agents[id].is_leader and Agents[id].location != 'Abroad':
            Agents[id].indirect_check(G,loc_dic[Agents[id].location].name,current_date)

        if loc_dic[Agents[id].location]:
            
            country = loc_dic[Agents[id].location].country

        else:
            country=None

        if Agents[id].status == 'Refugee':

            refugees+=frac
            if country == 'Burkina Faso':
                Burkinas+=frac
            elif country == "Côte D\'Ivoire":
                Cotes +=frac
            elif country == 'Guinea':
                Guineas +=frac
            elif country == 'Mauritania':
                Maus +=frac
            elif country == 'Niger':
                Nigers +=frac
            elif country == 'Senegal':
                Senegals +=frac
            elif Agents[id].location=='Abroad':
                Others += frac

    
        elif Agents[id].status == 'Returnee':
            returnees+=frac
            Malis+=frac
            
        elif Agents[id].status == 'Resident':
            residents+=frac
            Malis+=frac
            
        elif Agents[id].status == 'Dead':
            deaths+=frac

            
        elif Agents[id].status == 'IDP':
            idps+=frac
            Malis+=frac
        else:
            fleeing+=1
            Malis+=1
        
        if Agents[id].moved_today:
            
            
            
            if Agents[id].location!=Agents[id].shortterm and Agents[id].is_leader:
                
                for agent in Agents[id].group:
                    

                    if agent.location!=agent.shortterm:
                        try:
                            loc_dic[agent.shortterm].addmember(agent.id)
                            loc_dic[agent.location].removemember(agent.id,Agents)
                            G.nodes[agent.location]['population'] -= 1 # update nodes
                            G.nodes[agent.shortterm]['population'] += 1
                            G.edges[agent.location, agent.shortterm]['travelled']+=1
                        except:
                            print(agent.location)
                            pass

            if Agents[id].instratgroup:
                if Agents[id].moving:
                
                    if location=='Fassala':
                        if current_date<pd.Timestamp(datetime(2012, 3, 19).date()):
                            Agents[id].check_kick_out(ags)

                            if Agents[id].is_stuck:
                                Agents[id].speed_focus()
                            
                            if Agents[id].is_leader:
                                for loc_id in loc_dic[Agents[id].location].members:
                                    if loc_id!=id:
                                        if Agents[loc_id].is_leader:
                                            if Agents[loc_id].instratgroup:
                                                if not Agents[loc_id].comb:
                                                    if Agents[id].longterm==Agents[loc_id].longterm:
                                                        if Agents[loc_id].moving:
                                                            if Agents[loc_id].shortterm==Agents[id].shortterm:
                                                                Agents[id].super_imp(Agents[loc_id])
                                                                Agents[id].comb=True
                                                                Agents[loc_id].comb=True

                    else:
                        Agents[id].check_kick_out(ags)

                        if Agents[id].is_stuck:
                            Agents[id].speed_focus()
                        
                        if Agents[id].is_leader:
                            for loc_id in loc_dic[Agents[id].location].members:
                                if Agents[id].is_leader:
                                    for loc_id in loc_dic[Agents[id].location].members:
                                        if loc_id!=id:
                                            if Agents[loc_id].is_leader:
                                                if Agents[loc_id].instratgroup:
                                                    if not Agents[loc_id].comb:
                                                        if Agents[id].longterm==Agents[loc_id].longterm:
                                                            if Agents[loc_id].moving:
                                                                if Agents[loc_id].shortterm==Agents[id].shortterm:
                                                                    Agents[id].super_imp(Agents[loc_id])
                                                                    Agents[id].comb=True
                                                                    Agents[loc_id].comb=True
                        

            Agents[id].moved_today=False

        for id in Agents:
            Agents[id].comb=False
        
        
        processed_agents += 1  # Update the counter after processing each agent
        print_progress_bar(processed_agents, total_agents, prefix='Progress:', suffix='Complete', length=50)

    if current_date > pd.Timestamp(datetime(2012, 3, 19).date()):
        for agent in fassala.members:
            fassala.removemember(agent,Agents)
            mbera.addmember(agent)
            Agents[agent].location='Mbera'

    for camp in camps:
        populations[camp.name].append(frac*len(camp.members))

    statuses['Refugee'].append(refugees)
    statuses['Returnee'].append(returnees)
    statuses['Resident'].append(residents)
    statuses['Dead'].append(deaths)
    statuses['IDP'].append(idps)
    statuses['Fleeing from conflict'].append(fleeing)

    countries['Burkina Faso'].append(Burkinas)
    countries['Côte D\'Ivoire'].append(Cotes)
    countries['Guinea'].append(Guineas)
    countries['Mauritania'].append(Maus)
    countries['Niger'].append(Nigers)
    countries['Senegal'].append(Senegals)
    countries['Mali'].append(Malis)
    countries['Other'].append(Others)
    countries['Dead'].append(deaths)
    
    
    if bpd:
        for new_id in range(i, i + birth_rate + 1):
            if try_to_create_agent(new_id, birth_rate, loc_dic, Agents):
                update_location_and_groups(new_id, loc_dic, Agents, total_agents, ags)
        i += birth_rate
        total_agents+=birth_rate

    else:
        if days % birth_rate == 0:
            if try_to_create_agent(i, birth_rate, loc_dic, Agents):
                update_location_and_groups(i, loc_dic, Agents, total_agents, ags)
            i += 1
            total_agents+=1

    days+=1


end_time = time.time()
elapsed_time = end_time - start_time

print('\n')
print(f"Simulation completed in {elapsed_time:.2f} seconds.")

camp_names = list(populations.keys())
n_camps = len(camp_names)
camps_per_figure = 3

# Loop through the camps in chunks of 3
for i in range(0, n_camps, camps_per_figure):
    fig, axes = plt.subplots(nrows=1, ncols=camps_per_figure, figsize=(15, 5))
    
    for j in range(camps_per_figure):
        camp_index = i + j
        if camp_index < n_camps:  # Check to avoid index out of range
            ax = axes[j]
            camp_name = camp_names[camp_index]
            ax.plot(dates, populations[camp_name], label=camp_name) # adjusted for fraction of population
            ax.set_title(camp_name)
            ax.set_xlabel('Date')
            ax.set_ylabel('Population')
            ax.tick_params(axis='x', rotation=45)
        else:
            axes[j].axis('off')  # Hide unused subplot
    
    plt.tight_layout()
    plt.show()

csv_file = 'Camp_splits.csv'

populations['Date']=dates
statuses['Date']=dates
countries['Date']=dates

# Write the dictionary to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=populations.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for i in range(len(populations['Date'])):
        row = {key: populations[key][i] for key in populations.keys()}
        writer.writerow(row)


for camp in camps:
    print("Camp in " + str(camp.name) + " is " + str(camp.population*frac))

draw_graph(G, current_date, distances_on=False)

csv_file2= 'Status_splits.csv'

with open(csv_file2, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=statuses.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data row
    for i in range(len(statuses['Date'])):
        row = {key: statuses[key][i] for key in statuses.keys()}
        writer.writerow(row)

csv_file3= 'Country_split_refugees.csv'

with open(csv_file3, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=countries.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data row
    for i in range(len(countries['Date'])):
        row = {key: countries[key][i] for key in countries.keys()}
        writer.writerow(row)
