"""

TO DO

ALPHA 5

- Capacity empirically derived - No idea how to get this - assumed to be 20,000 as none of camps go above this


~4hr
- self.is_leader === >logic that randomly assigns leader
- Formation of groups
- Families will share the following:
----- 'No-Go' zones  
----- Slowest walking speed
- Followers (such as children) should not follow same decision logic (if statement at start which sees if follower)

~5hr
- Consider group size distributions and begin to model stategic groups
- Logic for reassessing at each node the group

~1hr
- Consider next to nodes (indirect) from conflicts as an attribute for cities (will help to model IDPs)

~2hr
- "DANGER LEVELS' -> roulette function now a function of a score rather than just distance

~4hr
- Conflict fatalieis -> agent death mechanisms
- Agent birth mechanisms
- Empirically derive births

~2hr
- Incorporate average times at different nodes

~3hr
- Foreign conflicts

VISUALISATION:
- Refugees/IDPs/Returnees distribution

SCORE BASED ON:
- Destination node distance
- How frequent the link is travelled
- Destination node danger
- First node in path danger
- What people are saying (i.e. other people's destination)
- Perhaps percentage full


COMMUNICATION
- Don't focus as much on the knowledge of routes and settlements
- Focus on the perception of danger
- i.e. an agent communicating with another agent about a dange level of node/link
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import time
from Locations import City, Camp, total_pop
from Agents import Agent
from Network import create_graph, draw_graph
from Visualisations import colors, print_progress_bar

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




# Define camps
bobo = Camp("Bobo", "Burkina Faso", population=0) # 2012-02-29
goudoubo = Camp("Goudoubo", "Burkina Faso",population=0) # 2012-02-29
mentao = Camp("Mentao", "Burkina Faso", population=0) # 2012-02-29
ouagadougou = Camp("Ouagadougou", "Burkina Faso", population=0) # 2012-02-29
fassala = Camp("Fassala", "Mauritania", population= 17790) # 2012-02-29
mbera = Camp("Mbera", "Mauritania", population=0) # 2012-02-29
abala = Camp("Abala", "Niger", population=1175) # 2012-02-29
intikane = Camp("Intikane", "Niger", population=5058) # 2013-05-07
mangaize = Camp("Mangaize", "Niger", population=1110) # 2012-02-29
niamey = Camp("Niamey", "Niger", population=0) # 2012-02-29
tabareybarey = Camp("Tabareybarey", "Niger", population=0) # 2012-02-29

camps = [bobo, goudoubo, mentao, ouagadougou, fassala, mbera, abala, intikane, mangaize, niamey, tabareybarey]

# populations from 2009 census https://www.citypopulation.de/en/mali/cities/

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


## Following section has been completed manually from inspection

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

# Calculate the number of days in 2012
start_date = pd.to_datetime('2012-01-01')
end_date = pd.to_datetime('2012-12-31')
dates = pd.date_range(start_date, end_date)

conflicts = pd.read_csv("1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")[
    pd.read_csv("1997-01-01-2024-03-01-Eastern_Africa-Middle_Africa-Northern_Africa-Southern_Africa-Western_Africa-Mali.csv")['year']==2012]

conflicts['event_date'] = pd.to_datetime(conflicts['event_date'], format="%d %B %Y")
conflicts.sort_values(by='event_date', inplace=True)

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
           'Kéniéba':kenieba}

# Locations without an OSMNX are assumed to be within the closest city

kidals = ['Imezzehene','Tin Kaloumane']
bamakos = ['Doubabougou','Kati','Kambila']

print("Starting simulation...")
start_time = time.time()




total_population = total_pop(cities)

#########################################################################################
frac = 1000 # TO VARY
#########################################################################################

n_agents = int(total_population/frac)

for loc in locations:
    loc.population= int(loc.population/frac)

for camp in camps:
    camp.capacity = int(camp.capacity/frac)

city_probabilities = {city.name: city.population / total_population for city in cities}

prob_values = list(city_probabilities.values())

# Normalize the probabilities
normalized_prob_values = [float(i)/sum(prob_values) for i in prob_values]

if not np.isclose(sum(normalized_prob_values), 1.0):
    raise ValueError("Normalized probabilities do not sum closely to 1.")

G = create_graph(locations)
ongoing_conflicts = []

Agent.calculate_distributions()
Agent.initialise_cities(foreign_cities)
Agent.initialise_populations(cities,total_population)

Agents = {}
ags = []
for i in list(range(1,n_agents+1)):
    Agents[i] = Agent(i)
    loc_dic[Agents[i].location].addmember(i)
    ags.append(Agents[i])




populations = {camp.name: [] for camp in camps}

total_agents = len(Agents)

for current_date in dates: 

    """
    
    This is where the model will be ran in real time
    
    """   
    processed_agents = 0
    

    print("\n")
    print(f"Simulating day: {current_date.strftime('%Y-%m-%d')}")

    for ongoing_conflict in ongoing_conflicts:
        ongoing_conflict.check_and_update_conflict_status(current_date) # check ongoing conflicts
        if ongoing_conflict.hasconflict:
            pass
        else:
            ongoing_conflicts.remove(ongoing_conflict) # remove conflict from stored list
            G.nodes[ongoing_conflict.name]['has_conflict']=False # update graph node

    for idx, event in conflicts[conflicts['event_date'] == current_date].iterrows(): # consider all conflicts in given day

        location = loc_dic[event['location']]

        # print("Conflict in " + event['location'])

        # Dealing with the exceptions between ACLED and OSMNX
        if event['location'] in kidals:
            G.nodes['Kidal']['has_conflict']=True
        elif event['location'] in bamakos:
            G.nodes['Bamako']['has_conflict']=True
        elif event['location']=='Tin Zaouaten':
            G.nodes['Tinzaouaten']['has_conflict']=True
        elif event['location']=='Komeye Koukou':
            G.nodes['Koue']['has_conflict']=True
        elif event['location']=='Syama Gold Mine':
            G.nodes['Syama']['has_conflict']=True
        else:
            G.nodes[event['location']]['has_conflict']=True
        location.in_city_conflict(event['fatalities'], current_date)
        if location not in ongoing_conflicts:
            ongoing_conflicts.append(location)

    # represent each epoch as a graph, commented out for now due to large number of unneccesary graphs.
    # draw_graph(G, current_date,ongoing_conflicts)

    # print(colors.PURPLE, "current conflicts",[con.name for con in ongoing_conflicts], colors.END)
    
    for id in Agents:
        Agents[id].assess_situation_and_move_if_needed(G,loc_dic[Agents[id].location],current_date)
        loc_dic[Agents[id].shortterm].addmember(id)
        loc_dic[Agents[id].location].removemember(id)
        G.nodes[Agents[id].location]['population'] -= 1 # update nodes
        G.nodes[Agents[id].shortterm]['population'] += 1

        Agents[id].merge_nogo_lists(ags) # allows nogo lists to be unionised

        processed_agents += 1  # Update the counter after processing each agent
        print_progress_bar(processed_agents, total_agents, prefix='Progress:', suffix='Complete', length=50)

    for camp in camps:
        pop=camp.population*frac
        populations[camp.name].append(pop)
    
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

csv_file = 'Simulated_pop_data.csv'

populations['Date']=dates

# Write the dictionary to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=populations.keys())
    
    # Write header
    writer.writeheader()
    
    # Write data rows
    for i in range(len(populations['Date'])):
        row = {key: populations[key][i] for key in populations.keys()}
        writer.writerow(row)

end_time = time.time()


elapsed_time = end_time - start_time

print(f"Simulation completed in {elapsed_time:.2f} seconds.")

"""
# e.g. where can an agent access within a day, starting at Bamako and travelling at 200km/d
accessible = find_accessible_nodes_within_distance(G, 'Bamako', 200)

print('Walking from Bamako:',accessible)

# e.g. how do I get to all camps from Bamako (shortest distance) (max move speed of 200)

camp_paths_from_bamako=camp_paths(G, 'Bamako',200)


if camp_paths_from_bamako is not None:
    for key in camp_paths_from_bamako:
        print('To get to %s from Bamako it takes %.1fkm and you travel through: %s' % (key, camp_paths_from_bamako[key]['distance'], camp_paths_from_bamako[key]['path']))


# e.g. how do I get to the nearest airport from Zegoua (max move speed of 200)
    
air_path = find_nearest_city_with_airport_directly(G, 'Bamako',200) 

print(air_path)

# How do I get to a bordering country e.g. Senegal (max move speed of 200)

count_path = find_shortest_paths_to_neighboring_countries(G, 'Bamako',200)

if count_path is not None:
    print(count_path)
"""

for camp in camps:
    print("Camp in " + str(camp.name) + " is " + str(camp.population))

for agent in ags:
    print(agent.nogos)


"""
for id in Agents:
    if Agents[id].capitalbracket == 'Rich':
        print(colors.GREEN + Agents[id].location + colors.END)
    elif Agents[id].capitalbracket == 'Mid':
        print(colors.YELLOW + Agents[id].location + colors.END)
    else:
        print(colors.RED + Agents[id].location + colors.END)
"""
# draw_graph(G, current_date, distances_on=True)
        