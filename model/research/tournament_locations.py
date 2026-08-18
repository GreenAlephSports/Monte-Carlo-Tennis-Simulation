"""Static tournament -> (city, country, lat, lon) lookup for the ATP/WTA tournament name strings
found in the Kaggle historical dataset (data/atp_tennis.csv, data/wta_tennis.csv).

This is deliberately NOT a live geocoder. The Kaggle data spells the same physical tournament
differently across eras (title-sponsor changes, e.g. "Sony Ericsson Open" / "NASDAQ-100 Open" /
"Miami Open" are all the same event in Miami/Key Biscayne), so the real mapping problem is
alias resolution, not geocoding a clean tournament name.

ALIAS_RULES is an ordered list of (needle, canonical_key_or_None) pairs. resolve_location(name)
picks the LONGEST needle that appears as a substring of the raw tournament name (case-insensitive) -
sorting by needle length, rather than hand-ordering ~150 rules, is what keeps a specific alias
(e.g. "German Open Tennis Championships" -> Hamburg) from being shadowed by a shorter generic one
that happens to be a substring of it (e.g. "German Open" -> Berlin, the WTA event's later name).
A canonical_key of None is an explicit exclusion: some raw names genuinely can't be resolved to one
city (year-end championships and Masters Cups rotate host city; the ATP "Canadian Open"/"Rogers
Masters"/"Rogers Cup" alternates Toronto/Montreal by year with no city info in the name itself) -
these are listed so a later, broader rule doesn't accidentally swallow them, not because a match
was ever found for them.

Known, accepted limitations (see the coverage report the caller prints, not asserted here):
  - Lat/lon are city-center or well-known venue coordinates from general knowledge, not verified
    against an authoritative venue database - fine for daily-weather purposes (Open-Meteo's grid
    cells are ~9-25km) but not survey-grade.
  - A handful of tournaments genuinely moved city more than once within the dataset's span in a
    way not disambiguated by the raw name (e.g. Brasil Open, Masters Cup, WTA/ATP Finals, Copa
    Telmex, Ecuador Open) - deliberately excluded (mapped to None) rather than guessed.
  - Indoor tournaments are mapped like any other - a hot/cold-weather hypothesis test should
    probably restrict to Outdoor Court values (the Kaggle data's own Court column) rather than
    rely on this module to know which arenas are climate-controlled.
"""

# canonical_key -> (city, country, lat, lon)
LOCATIONS = {
    # Grand Slams
    "AUS_OPEN": ("Melbourne", "Australia", -37.8136, 144.9631),
    "FRENCH_OPEN": ("Paris", "France", 48.8566, 2.3522),
    "WIMBLEDON": ("London (Wimbledon)", "UK", 51.4344, -0.2135),
    "US_OPEN": ("New York (Flushing Meadows)", "USA", 40.7498, -73.8459),

    # Masters 1000 / Premier Mandatory
    "INDIAN_WELLS": ("Indian Wells, CA", "USA", 33.7175, -116.3016),
    "MIAMI": ("Miami/Key Biscayne, FL", "USA", 25.7617, -80.1918),
    "MONTE_CARLO": ("Monte Carlo", "Monaco", 43.7384, 7.4246),
    "MADRID": ("Madrid", "Spain", 40.4168, -3.7038),
    "ROME": ("Rome", "Italy", 41.9028, 12.4964),
    "TORONTO": ("Toronto", "Canada", 43.6532, -79.3832),
    "MONTREAL": ("Montreal", "Canada", 45.5019, -73.5674),
    "CINCINNATI": ("Mason (Cincinnati), OH", "USA", 39.3703, -84.3230),
    "SHANGHAI": ("Shanghai", "China", 31.2304, 121.4737),
    "PARIS_BERCY": ("Paris (Bercy)", "France", 48.8383, 2.3782),
    "BEIJING": ("Beijing", "China", 39.9042, 116.4074),
    "WUHAN": ("Wuhan", "China", 30.5928, 114.3055),

    # ATP/WTA 500, 250, Premier, International-level recurring stops
    "ROTTERDAM": ("Rotterdam", "Netherlands", 51.9244, 4.4777),
    "ACAPULCO": ("Acapulco", "Mexico", 16.8531, -99.8237),
    "MUNICH": ("Munich", "Germany", 48.1351, 11.5820),
    "CASABLANCA": ("Casablanca", "Morocco", 33.5731, -7.5898),
    "DOHA": ("Doha", "Qatar", 25.2854, 51.5310),
    "NEWPORT": ("Newport, RI", "USA", 41.4901, -71.3128),
    "STUTTGART": ("Stuttgart", "Germany", 48.7758, 9.1829),
    "NUREMBERG": ("Nuremberg", "Germany", 49.4521, 11.0767),
    "WASHINGTON_DC": ("Washington, DC", "USA", 38.9072, -77.0369),
    "BASEL": ("Basel", "Switzerland", 47.5596, 7.5886),
    "HALLE": ("Halle (Westf.)", "Germany", 51.9614, 8.3308),
    "ZAGREB": ("Zagreb", "Croatia", 45.8150, 15.9819),
    "VIENNA": ("Vienna", "Austria", 48.2082, 16.3738),
    "QUEENS_CLUB": ("London (Queen's Club)", "UK", 51.4875, -0.2143),
    "EASTBOURNE": ("Eastbourne", "UK", 50.7684, 0.2901),
    "NOTTINGHAM": ("Nottingham", "UK", 52.9548, -1.1581),
    "BIRMINGHAM": ("Birmingham", "UK", 52.4862, -1.8904),
    "BARCELONA": ("Barcelona", "Spain", 41.3888, 2.1590),
    "MONTPELLIER": ("Montpellier", "France", 43.6119, 3.8772),
    "DELRAY_BEACH": ("Delray Beach, FL", "USA", 26.4615, -80.0728),
    "TOKYO": ("Tokyo", "Japan", 35.6762, 139.6503),
    "RIO": ("Rio de Janeiro", "Brazil", -22.9068, -43.1729),
    "INDIANAPOLIS": ("Indianapolis, IN", "USA", 39.7684, -86.1581),
    "CHENNAI": ("Chennai", "India", 13.0827, 80.2707),
    "BUCHAREST": ("Bucharest", "Romania", 44.4268, 26.1025),
    "BUENOS_AIRES": ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    "BANGKOK": ("Bangkok", "Thailand", 13.7563, 100.5018),
    "LYON": ("Lyon", "France", 45.7640, 4.8357),
    "GENEVA": ("Geneva", "Switzerland", 46.2044, 6.1432),
    "STOCKHOLM": ("Stockholm", "Sweden", 59.3293, 18.0686),
    "BASTAD": ("Bastad", "Sweden", 56.4283, 12.8535),
    "MEMPHIS": ("Memphis, TN", "USA", 35.1495, -90.0490),
    "ATLANTA": ("Atlanta, GA", "USA", 33.7490, -84.3880),
    "MALLORCA": ("Mallorca", "Spain", 39.5696, 2.6502),
    "SOPOT": ("Sopot", "Poland", 54.4418, 18.5601),
    "PALERMO": ("Palermo", "Italy", 38.1157, 13.3615),
    "VALENCIA": ("Valencia", "Spain", 39.4699, -0.3763),
    "CHENGDU": ("Chengdu", "China", 30.5728, 104.0668),
    "LOS_ANGELES": ("Los Angeles, CA", "USA", 34.0522, -118.2437),
    "SAN_JOSE": ("San Jose, CA", "USA", 37.3382, -121.8863),
    "HAMBURG": ("Hamburg", "Germany", 53.5511, 9.9937),
    "BERLIN": ("Berlin", "Germany", 52.5200, 13.4050),
    "ST_PETERSBURG": ("St. Petersburg", "Russia", 59.9311, 30.3609),
    "MOSCOW": ("Moscow", "Russia", 55.7558, 37.6173),
    "METZ": ("Metz", "France", 49.1193, 6.1757),
    "WINSTON_SALEM": ("Winston-Salem, NC", "USA", 36.0999, -80.2442),
    "DUBAI": ("Dubai", "UAE", 25.2048, 55.2708),
    "AUCKLAND": ("Auckland", "New Zealand", -36.8485, 174.7633),
    "GSTAAD": ("Gstaad", "Switzerland", 46.4739, 7.2870),
    "VINA_DEL_MAR": ("Vina del Mar", "Chile", -33.0246, -71.5518),
    "CORDOBA": ("Cordoba", "Argentina", -31.4201, -64.1888),
    "BRISBANE": ("Brisbane", "Australia", -27.4698, 153.0251),
    "UMAG": ("Umag", "Croatia", 45.4353, 13.5253),
    "ESTORIL": ("Estoril/Cascais", "Portugal", 38.7071, -9.3980),
    "BELGRADE": ("Belgrade", "Serbia", 44.7866, 20.4489),
    "SYDNEY": ("Sydney", "Australia", -33.8688, 151.2093),
    "HOUSTON": ("Houston, TX", "USA", 29.7604, -95.3698),
    "BOGOTA": ("Bogota", "Colombia", 4.7110, -74.0721),
    "CHARLESTON": ("Charleston, SC", "USA", 32.7765, -79.9311),
    "STRASBOURG": ("Strasbourg", "France", 48.5734, 7.7521),
    "FES": ("Fes", "Morocco", 34.0181, -5.0078),
    "TASHKENT": ("Tashkent", "Uzbekistan", 41.2995, 69.2401),
    "HOBART": ("Hobart", "Australia", -42.8821, 147.3272),
    "PATTAYA": ("Pattaya", "Thailand", 12.9236, 100.8825),
    "LUXEMBOURG": ("Luxembourg City", "Luxembourg", 49.6116, 6.1319),
    "SEOUL": ("Seoul", "South Korea", 37.5665, 126.9780),
    "LINZ": ("Linz", "Austria", 48.3069, 14.2858),
    "GUANGZHOU": ("Guangzhou", "China", 23.1291, 113.2644),
    "ISTANBUL": ("Istanbul", "Turkey", 41.0082, 28.9784),
    "BAKU": ("Baku", "Azerbaijan", 40.4093, 49.8671),
    "S_HERTOGENBOSCH": ("s-Hertogenbosch", "Netherlands", 51.6978, 5.3037),
    "PRAGUE": ("Prague", "Czechia", 50.0755, 14.4378),
    "BAD_HOMBURG": ("Bad Homburg", "Germany", 50.2266, 8.6181),
    "TIANJIN": ("Tianjin", "China", 39.3434, 117.3616),
    "STANFORD": ("Stanford, CA", "USA", 37.4275, -122.1697),
    "SAN_DIEGO": ("San Diego, CA", "USA", 32.7157, -117.1611),
    "MERIDA": ("Merida", "Mexico", 20.9674, -89.5926),
    "MARBELLA": ("Marbella", "Spain", 36.5099, -4.8863),
    "BUDAPEST": ("Budapest", "Hungary", 47.4979, 19.0402),
    "OSTRAVA": ("Ostrava", "Czechia", 49.8209, 18.2625),
    "NEW_HAVEN": ("New Haven, CT", "USA", 41.3083, -72.9279),
    "ADELAIDE": ("Adelaide", "Australia", -34.9285, 138.6007),
    "SHENZHEN": ("Shenzhen", "China", 22.5431, 114.0579),
    "HONG_KONG": ("Hong Kong", "China", 22.3193, 114.1694),
    "MARSEILLE": ("Marseille", "France", 43.2965, 5.3698),
    "NICE": ("Nice", "France", 43.7102, 7.2620),
    "SINGAPORE": ("Singapore", "Singapore", 1.3521, 103.8198),
    "WARSAW": ("Warsaw", "Poland", 52.2297, 21.0122),
    "KUALA_LUMPUR": ("Kuala Lumpur", "Malaysia", 3.1390, 101.6869),
    "SOFIA": ("Sofia", "Bulgaria", 42.6977, 23.3219),
    "ASTANA": ("Astana", "Kazakhstan", 51.1605, 71.4704),
    "ANTALYA": ("Antalya", "Turkey", 36.8969, 30.7133),
    "TALLINN": ("Tallinn", "Estonia", 59.4370, 24.7536),
    "KATOWICE": ("Katowice", "Poland", 50.2649, 19.0238),
    "ZURICH": ("Zurich", "Switzerland", 47.3769, 8.5417),
    "LAUSANNE": ("Lausanne", "Switzerland", 46.5197, 6.6323),
    "MONTERREY": ("Monterrey", "Mexico", 25.6866, -100.3161),
    "GUADALAJARA": ("Guadalajara", "Mexico", 20.6597, -103.3496),
    "ABU_DHABI": ("Abu Dhabi", "UAE", 24.4539, 54.3773),
    "KITZBUHEL": ("Kitzbuhel", "Austria", 47.4467, 12.3927),
    "BAD_GASTEIN": ("Bad Gastein", "Austria", 47.1147, 13.1322),
    "QUEBEC_CITY": ("Quebec City", "Canada", 46.8139, -71.2080),
    "CLUJ_NAPOCA": ("Cluj-Napoca", "Romania", 46.7712, 23.6236),
    "COPENHAGEN": ("Copenhagen", "Denmark", 55.6761, 12.5683),
    "DALLAS": ("Dallas, TX", "USA", 32.7767, -96.7970),
    "LOS_CABOS": ("Los Cabos", "Mexico", 22.8905, -109.9167),
    "NANCHANG": ("Nanchang", "China", 28.6820, 115.8579),
    "AMELIA_ISLAND": ("Amelia Island, FL", "USA", 30.6699, -81.4437),
    "ROUEN": ("Rouen", "France", 49.4432, 1.0993),
    "TAIPEI": ("Taipei", "Taiwan", 25.0330, 121.5654),
    "PORTOROZ": ("Portoroz", "Slovenia", 45.5150, 13.5967),
    "AUSTIN": ("Austin, TX", "USA", 30.2672, -97.7431),
    "CLEVELAND": ("Cleveland, OH", "USA", 41.4993, -81.6944),
    "BRUSSELS": ("Brussels", "Belgium", 50.8503, 4.3517),
    "NINGBO": ("Ningbo", "China", 29.8683, 121.5440),
    "IASI": ("Iasi", "Romania", 47.1585, 27.6014),
}

# Ordered (needle, canonical_key_or_None) pairs. Matching sorts by len(needle) descending, so more
# specific needles always win over a shorter needle that happens to be their substring - list order
# here doesn't matter for correctness, only readability (grouped roughly by location).
ALIAS_RULES = [
    ("Australian Open", "AUS_OPEN"),
    ("French Open", "FRENCH_OPEN"),
    ("Roland Garros", "FRENCH_OPEN"),
    ("Wimbledon", "WIMBLEDON"),
    ("US Open", "US_OPEN"),

    ("BNP Paribas Open", "INDIAN_WELLS"),
    ("Pacific Life Open", "INDIAN_WELLS"),
    ("Indian Wells", "INDIAN_WELLS"),
    ("Newsweek Champions Cup", "INDIAN_WELLS"),

    ("Sony Ericsson Open", "MIAMI"),
    ("NASDAQ-100 Open", "MIAMI"),
    ("Ericsson Open", "MIAMI"),
    ("Lipton", "MIAMI"),
    ("Miami Open", "MIAMI"),

    ("Monte Carlo Masters", "MONTE_CARLO"),

    ("Mutua Madrid Open", "MADRID"),
    ("Madrile", "MADRID"),  # covers "Mutua Madrileña Madrid Open" without spanning the accented char
    ("Madrid Masters", "MADRID"),

    ("Internazionali BNL d'Italia", "ROME"),
    ("Telecom Italia Masters Roma", "ROME"),
    ("Campionati Internazionali d'Italia", "ROME"),
    ("Rome TMS", "ROME"),

    ("Toronto TMS", "TORONTO"),
    ("Toronto", "TORONTO"),
    ("Montreal TMS", "MONTREAL"),
    ("Montreal", "MONTREAL"),
    # generic, city-unspecified Canadian Masters/Open names alternate Toronto/Montreal by year -
    # cannot be resolved from the name alone, deliberately excluded rather than guessed
    ("Rogers Masters", None),
    ("Rogers Cup", None),
    ("Canadian Open", None),

    ("Western & Southern Financial Group Masters", "CINCINNATI"),
    ("Western & Southern Financial Group Women's Open", "CINCINNATI"),
    ("Western & Southern Open", "CINCINNATI"),
    ("Cincinnati TMS", "CINCINNATI"),
    ("Cincinnati", "CINCINNATI"),

    ("Shanghai Masters", "SHANGHAI"),
    ("Heineken Open Shanghai", "SHANGHAI"),
    ("Shanghai", "SHANGHAI"),

    ("BNP Paribas Masters", "PARIS_BERCY"),

    ("China Open", "BEIJING"),
    ("Beijing", "BEIJING"),

    ("Wuhan Open", "WUHAN"),

    ("ABN AMRO World Tennis Tournament", "ROTTERDAM"),
    ("Rotterdam", "ROTTERDAM"),

    ("Abierto Mexicano Mifel", "ACAPULCO"),
    ("Abierto Mexicano", "ACAPULCO"),
    ("Copa AT&T", "ACAPULCO"),

    ("BMW Open", "MUNICH"),

    ("Grand Prix Hassan II", "CASABLANCA"),

    ("Qatar Exxon Mobil Open", "DOHA"),
    ("Qatar Total Open", "DOHA"),
    ("Qatar Ladies Open", "DOHA"),
    ("Qatar Open", "DOHA"),

    ("Hall of Fame Championships", "NEWPORT"),

    ("Mercedes Cup", "STUTTGART"),
    ("Porsche Tennis Grand Prix", "STUTTGART"),
    ("Porsche Grand Prix", "STUTTGART"),

    ("Versicherungscup", "NUREMBERG"),  # covers "Nürnberger Versicherungscup" without spanning ü
    ("Generali Ladies Linz Open", "LINZ"),  # placed before generic Nuremberg-esque "Linz" below

    ("Legg Mason Classic", "WASHINGTON_DC"),
    ("Citi Open", "WASHINGTON_DC"),

    ("Swiss Indoors", "BASEL"),
    ("Davidoff Swiss Indoors", "BASEL"),

    ("Gerry Weber Open", "HALLE"),
    ("Halle Open", "HALLE"),

    ("PBZ Zagreb Indoors", "ZAGREB"),

    ("Erste Bank Open", "VIENNA"),
    ("BA-CA Tennis Trophy", "VIENNA"),
    ("Bet-At-Home Cup", "VIENNA"),
    ("bet-at-home Open", "VIENNA"),
    ("Vienna Open", "VIENNA"),

    ("AEGON Championships", "QUEENS_CLUB"),
    ("Stella Artois", "QUEENS_CLUB"),
    ("Queen's Club Championships", "QUEENS_CLUB"),
    ("Queens Club", "QUEENS_CLUB"),

    ("Rothesay International", "EASTBOURNE"),
    ("AEGON International", "EASTBOURNE"),
    ("Eastbourne International", "EASTBOURNE"),

    ("The Nottingham Open", "NOTTINGHAM"),
    ("Nottingham Open", "NOTTINGHAM"),

    ("Rothesay Classic", "BIRMINGHAM"),
    ("Birmingham Classic", "BIRMINGHAM"),
    ("AEGON Classic", "BIRMINGHAM"),

    ("Open Seat Godo", "BARCELONA"),
    ("Open Banco Sabadell", "BARCELONA"),
    ("Barcelona Ladies Open", "BARCELONA"),
    ("Barcelona KIA", "BARCELONA"),
    ("Barcelona Open", "BARCELONA"),

    ("Open Sud de France", "MONTPELLIER"),

    ("Delray Beach Open", "DELRAY_BEACH"),

    ("Rakuten Japan Open Tennis Championships", "TOKYO"),
    ("Japan Open Tennis Championships", "TOKYO"),
    ("AIG Japan Open Tennis Championships", "TOKYO"),
    ("Toray Pan Pacific Open Tennis Tournament", "TOKYO"),
    ("Toray Pan Pacific Open", "TOKYO"),
    ("Japan Women's Tennis Open", "TOKYO"),
    ("Japan Women's Open Tennis", "TOKYO"),
    ("Japan Open", "TOKYO"),

    ("Rio Open", "RIO"),

    ("RCA Championships", "INDIANAPOLIS"),
    ("Indianapolis Tennis Championships", "INDIANAPOLIS"),

    ("Chennai Open", "CHENNAI"),

    ("Open Romania", "BUCHAREST"),
    ("BRD Nastase Tiriac Trophy", "BUCHAREST"),
    ("Bucharest Open", "BUCHAREST"),

    ("Argentina Open", "BUENOS_AIRES"),
    ("ATP Buenos Aires", "BUENOS_AIRES"),

    ("Thailand Open 2", None),  # sometimes used for a distinct, unspecified second event
    ("Thailand Open", "BANGKOK"),

    ("Grand Prix de Lyon", "LYON"),
    ("Lyon Open", "LYON"),

    ("Geneva Open", "GENEVA"),

    ("Stockholm Open", "STOCKHOLM"),

    ("SkiStar Swedish Open", "BASTAD"),
    ("Catella Swedish Open", "BASTAD"),
    ("Collector Swedish Open", "BASTAD"),
    ("Sony Swedish Open", "BASTAD"),
    ("Synsam Swedish Open", "BASTAD"),
    ("Nordea Open", "BASTAD"),
    ("Swedish Open", "BASTAD"),

    ("Regions Morgan Keegan Championships & the Cellular South Cup", "MEMPHIS"),
    ("Regions Morgan Keegan Championships", "MEMPHIS"),
    ("Cellular South Cup", "MEMPHIS"),
    ("Kroger St. Jude", "MEMPHIS"),
    ("U.S. National Indoor Tennis Championships", "MEMPHIS"),
    ("US National Indoor Tennis Championships", "MEMPHIS"),
    ("Memphis International", "MEMPHIS"),
    ("Memphis Classic", "MEMPHIS"),
    ("Memphis Open", "MEMPHIS"),

    ("BB&T Atlanta Open", "ATLANTA"),
    ("Atlanta Tennis Championships", "ATLANTA"),
    ("Atlanta Open", "ATLANTA"),

    ("Mallorca Championships", "MALLORCA"),
    ("Mallorca Open", "MALLORCA"),

    ("Idea Prokom Open", "SOPOT"),
    ("Orange Prokom Open", "SOPOT"),

    ("Campionati Internazionali Di Sicilia", "PALERMO"),
    ("Internazionali Femminili di Tennis di Palermo", "PALERMO"),
    ("Internazionali Femminili di Palermo", "PALERMO"),

    ("Valencia Open 500", "VALENCIA"),
    ("Open de Tenis Comunidad Valenciana", "VALENCIA"),
    ("CAM Open Comunidad Valenciana", "VALENCIA"),
    ("Open Sabadell Atlantico 2008", "VALENCIA"),

    ("Chengdu Open", "CHENGDU"),

    ("Mercedes-Benz Cup", "LOS_ANGELES"),
    ("Countrywide Classic", "LOS_ANGELES"),
    ("Farmers Classic", "LOS_ANGELES"),
    ("LA Tennis Open", "LOS_ANGELES"),
    ("LA Womens Tennis Championships", "LOS_ANGELES"),

    ("SAP Open", "SAN_JOSE"),
    ("Siebel Open", "SAN_JOSE"),
    ("Sybase Open", "SAN_JOSE"),

    ("Hamburg TMS", "HAMBURG"),
    ("Hamburg European Open", "HAMBURG"),
    ("German Open Tennis Championships", "HAMBURG"),
    ("International German Open", "HAMBURG"),
    ("Hamburg Open", "HAMBURG"),

    ("Qatar Telecom German Open", "BERLIN"),
    ("German Open", "BERLIN"),

    ("St. Petersburg Open", "ST_PETERSBURG"),

    ("Kremlin Cup", "MOSCOW"),
    ("Moscow River Cup", "MOSCOW"),

    ("Open de Moselle", "METZ"),

    ("Winston-Salem Open at Wake Forest University", "WINSTON_SALEM"),

    ("Dubai Duty Free Tennis Championships", "DUBAI"),
    ("Barclays Dubai Tennis Championships", "DUBAI"),
    ("Dubai Duty Free Men's Open", "DUBAI"),
    ("Dubai Duty Free Women's Open", "DUBAI"),
    ("Dubai Tennis Championships", "DUBAI"),
    ("Dubai Championships", "DUBAI"),
    ("Dubai Open", "DUBAI"),

    ("Heineken Trophy", "AUCKLAND"),
    ("Heineken Open", "AUCKLAND"),
    ("ASB Classic", "AUCKLAND"),

    ("Crédit Agricole Suisse Open Gstaad", "GSTAAD"),
    ("Suisse Open Gstaad", "GSTAAD"),
    ("Ladies Championship Gstaad", "GSTAAD"),
    ("Gstaad Open", "GSTAAD"),

    ("Royal Guard Open Chile", "VINA_DEL_MAR"),
    ("Movistar Open", "VINA_DEL_MAR"),
    ("VTR Open", "VINA_DEL_MAR"),
    ("Chile Open", "VINA_DEL_MAR"),

    ("Cordoba Open", "CORDOBA"),

    ("Brisbane International", "BRISBANE"),

    ("Studena Croatia Open", "UMAG"),
    ("ATP Vegeta Croatia Open", "UMAG"),
    ("Konzum Croatia Open", "UMAG"),
    ("Croatia Open", "UMAG"),

    ("Millennium Estoril Open", "ESTORIL"),
    ("Millenium Estoril Open", "ESTORIL"),
    ("Estoril Open", "ESTORIL"),

    ("Serbia Open", "BELGRADE"),
    ("Serbia Ladies Open", "BELGRADE"),
    ("Srpska Open", "BELGRADE"),
    ("Belgrade Open", "BELGRADE"),

    ("Apia International", "SYDNEY"),
    ("adidas International", "SYDNEY"),
    ("Medibank International", "SYDNEY"),
    ("Sydney Tennis Classic", "SYDNEY"),
    ("Sydney International", "SYDNEY"),

    ("Franklin Templeton Tennis Classic", "HOUSTON"),
    ("U.S. Men's Clay Court Championships", "HOUSTON"),
    ("U.S.Men's Clay Court Championships", "HOUSTON"),
    ("U.S. Clay Court Championships", "HOUSTON"),

    ("Copa Sony Ericsson Colsanitas", "BOGOTA"),
    ("Copa BBVA Colsanitas", "BOGOTA"),
    ("Copa Claro Colsanitas", "BOGOTA"),
    ("Copa Colsanitas Santander", "BOGOTA"),
    ("Copa Colsanitas", "BOGOTA"),

    ("Family Circle Cup", "CHARLESTON"),
    ("Volvo Car Open", "CHARLESTON"),
    ("Charleston Open", "CHARLESTON"),

    ("Internationaux de Strasbourg", "STRASBOURG"),

    ("FesGrand Prix de SAR La Princesse Lalla Meryem", "FES"),
    ("Grand Prix de SAR La Princesse Lalla Meryem", "FES"),
    ("Grand Prix SAR Lalla Meryem", "FES"),
    ("Morocco Open", "FES"),

    ("Tashkent Open", "TASHKENT"),

    ("Moorilla Hobart International", "HOBART"),
    ("Hobart International", "HOBART"),

    ("Pattaya Women's Open", "PATTAYA"),

    ("BGL BNP Paribas Luxembourg Open", "LUXEMBOURG"),
    ("FORTIS Championships Luxembourg", "LUXEMBOURG"),

    ("Hansol Korea Open", "SEOUL"),
    ("Kia Korea Open", "SEOUL"),
    ("KDB Korea Open", "SEOUL"),
    ("Korea Open", "SEOUL"),

    ("Generali Ladies Linz", "LINZ"),
    ("Ladies Linz Open", "LINZ"),

    ("TOE Life Ceramics Guangzhou International Women's Open", "GUANGZHOU"),
    ("Landsky Lighting Guangzhou International Women's Open", "GUANGZHOU"),
    ("GDD-Guangzhou International Womens Open", "GUANGZHOU"),
    ("WANLIMA Guangzhou International Women's Open", "GUANGZHOU"),
    ("GRC Bank Guangzhou International Women's Open", "GUANGZHOU"),
    ("Guangzhou International Women's Open", "GUANGZHOU"),
    ("Guangzhou Open", "GUANGZHOU"),

    ("TEB BNP Paribas Istanbul Cup", "ISTANBUL"),
    ("Istanbul Cup", "ISTANBUL"),
    ("Istanbul Open", "ISTANBUL"),

    ("Baku Cup", "BAKU"),

    ("Rosmalen Grass Court Championships", "S_HERTOGENBOSCH"),
    ("Priority Telecom Dutch Open", "S_HERTOGENBOSCH"),
    ("Topshelf Open", "S_HERTOGENBOSCH"),
    ("Unicef Open", "S_HERTOGENBOSCH"),
    ("Ordina Open", "S_HERTOGENBOSCH"),
    ("Aegon Open", "S_HERTOGENBOSCH"),
    ("Dutch Open", "S_HERTOGENBOSCH"),

    ("J&T Banka Prague Open", "PRAGUE"),
    ("ECM Prague Open", "PRAGUE"),
    ("Prague Open", "PRAGUE"),

    ("Bad Homburg Open", "BAD_HOMBURG"),

    ("Tianjin Open", "TIANJIN"),

    ("Bank of the West Classic", "STANFORD"),
    ("Mubadala Silicon Valley Classic", "STANFORD"),

    ("East West Bank Classic", "SAN_DIEGO"),
    ("Mercury Insurance Open", "SAN_DIEGO"),
    ("Acura Classic", "SAN_DIEGO"),
    ("San Diego Open", "SAN_DIEGO"),

    ("Merida Open", "MERIDA"),

    ("Andalucia Tennis Experience", "MARBELLA"),

    ("Gaz de France Budapest Grand Prix", "BUDAPEST"),
    ("POLI-FARBE Budapest Grand Prix", "BUDAPEST"),
    ("Hungarian Grand Prix", "BUDAPEST"),
    ("Hungarian Ladies Open", "BUDAPEST"),
    ("Gazprom Hungarian Open", "BUDAPEST"),
    ("Budapest Open", "BUDAPEST"),

    ("Ostrava Open", "OSTRAVA"),

    ("New Haven Open at Yale", "NEW_HAVEN"),
    ("Connecticut Open", "NEW_HAVEN"),
    ("Pilot Pen Tennis", "NEW_HAVEN"),

    ("Next Generation Adelaide International", "ADELAIDE"),
    ("Adelaide International", "ADELAIDE"),

    ("Shenzhen Longgang Gemdale Open", "SHENZHEN"),
    ("Shenzhen Open", "SHENZHEN"),

    ("Prudential Hong Kong Tennis Open", "HONG_KONG"),
    ("Hong Kong Tennis Open", "HONG_KONG"),

    ("Open 13", "MARSEILLE"),
    ("Marseille Open", "MARSEILLE"),

    ("Open de Nice Côte d’Azur", "NICE"),
    ("Open de Nice", "NICE"),

    ("Singapore Open", "SINGAPORE"),

    ("Warsaw Open", "WARSAW"),

    ("Proton Malaysian Open", "KUALA_LUMPUR"),
    ("Malaysian Open", "KUALA_LUMPUR"),
    ("Malaysia Open", "KUALA_LUMPUR"),

    ("Garanti Koza Sofia Open", "SOFIA"),
    ("Qatar Airways Tournament of Champions Sofia", "SOFIA"),
    ("Sofia Open", "SOFIA"),

    ("Astana Open", "ASTANA"),

    ("Antalya Open", "ANTALYA"),

    ("Tallinn Open", "TALLINN"),

    ("BNP Paribas Katowice Open", "KATOWICE"),
    ("Katowice Open", "KATOWICE"),

    ("Zurich Open", "ZURICH"),

    ("Ladies Open Lausanne", "LAUSANNE"),

    ("Monterrey Open", "MONTERREY"),

    ("Abierto Zapopan", "GUADALAJARA"),
    ("Guadalajara Open", "GUADALAJARA"),

    ("Abu Dhabi WTA Women's Tennis Open", "ABU_DHABI"),

    ("Generali Open", "KITZBUHEL"),
    ("Internationaler Raiffeisen Grand Prix", "KITZBUHEL"),
    ("Austrian Open", "KITZBUHEL"),

    ("Gastein Ladies", "BAD_GASTEIN"),

    ("Bell Challenge", "QUEBEC_CITY"),
    ("Coupe Banque Nationale", "QUEBEC_CITY"),

    ("Transylvania Open", "CLUJ_NAPOCA"),

    ("Copenhagen Open", "COPENHAGEN"),

    ("Dallas Open", "DALLAS"),

    ("Los Cabos Open", "LOS_CABOS"),

    ("Jiangxi Women's Tennis Open", "NANCHANG"),
    ("Jiangxi Tennis Open", "NANCHANG"),
    ("Jiangxi Open", "NANCHANG"),

    ("Bausch & Lomb Championships", "AMELIA_ISLAND"),

    ("Open de Rouen", "ROUEN"),

    ("Taiwan Open", "TAIPEI"),

    ("Banka Koper Slovenia Open", "PORTOROZ"),
    ("Slovenia Open", "PORTOROZ"),

    ("ATX Open", "AUSTIN"),

    ("Tennis in the Land", "CLEVELAND"),

    ("Poland Open", "WARSAW"),

    ("TATA Open", "CHENNAI"),

    ("Stuttgart Open", "STUTTGART"),
    ("Stuttgart TMS", "STUTTGART"),

    ("German Tennis Championships", "HAMBURG"),

    ("AAPT Championships", "ADELAIDE"),

    ("Open Gaz de France", "PARIS_BERCY"),
    ("Open GDF Suez", "PARIS_BERCY"),

    ("Allianz Suisse Open", "GSTAAD"),
    ("CA Tennis Trophy", "VIENNA"),
    ("Brussels Open", "BRUSSELS"),
    ("Ningbo Open", "NINGBO"),
    ("Iasi Open", "IASI"),

    # explicit exclusions - rotating/ambiguous host city, no reliable resolution from the name
    ("Masters Cup", None),
    ("BNP Paribas WTA Finals", None),
    ("WTA Elite Trophy", None),
    ("WTA Finals", None),
    ("Sony Ericsson Championships", None),
    ("Garanti Koza WTA Tournament of Champions", None),
    ("Commonwealth Bank Tournament of Champions", None),
    ("Brasil Open", None),
    ("Brasil Tennis Cup", None),
    ("Copa Telmex", None),
    ("Copa Claro", None),
    ("Ecuador Open", None),
    ("International Championships", None),
    ("European Open", None),

    # bare "BNP Paribas" (old ATP Paris Masters listing with no further qualifier) - kept last so
    # every more specific BNP Paribas variant above (and the explicit WTA Finals exclusion) wins
    ("BNP Paribas", "PARIS_BERCY"),
]

_SORTED_RULES = sorted(ALIAS_RULES, key=lambda r: -len(r[0]))


def resolve_location(tournament_name):
    """Returns (canonical_key, city, country, lat, lon) for a raw Kaggle Tournament string, or
    None if it doesn't match any known alias (never seen before) or matches an explicit
    exclusion (a real tournament whose host city can't be determined from the name alone)."""
    if not isinstance(tournament_name, str):
        return None
    name_lower = tournament_name.lower()
    for needle, canonical_key in _SORTED_RULES:
        if needle.lower() in name_lower:
            if canonical_key is None:
                return None
            city, country, lat, lon = LOCATIONS[canonical_key]
            return canonical_key, city, country, lat, lon
    return None
