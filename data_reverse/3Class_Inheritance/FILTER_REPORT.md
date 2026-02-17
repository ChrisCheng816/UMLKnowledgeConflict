继承关系三元组过滤与补全报告

总行数（过滤前）：1932
总行数（过滤后+补全）：1932

各子目录统计：
| 子目录(data.txt) | 过滤前 | 过滤后 | 删除 | 新增 |
|---|---:|---:|---:|---:|
| Animal | 135 | 135 | 0 | 0 |
| Car | 77 | 77 | 43 | 43 |
| Countries | 138 | 138 | 25 | 25 |
| Entertainment | 450 | 450 | 0 | 0 |
| Food | 274 | 274 | 0 | 0 |
| Items | 30 | 30 | 0 | 0 |
| People | 34 | 34 | 0 | 0 |
| Place | 89 | 89 | 1 | 1 |
| Plant | 108 | 108 | 0 | 0 |
| Sport | 234 | 234 | 0 | 0 |
| Universities | 363 | 363 | 0 | 0 |

新增数据约束检查：
- 本次新增三元组数量：69
- 新增数据中 (b,c) 唯一性：通过（共 69 个唯一 (b,c)）

## 删除项：Car/data.txt
- 行35: ToyotaCorolla Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行36: ToyotaPrius Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行37: ToyotaRAV4 Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行38: ToyotaHighlander Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行39: ToyotaTacoma Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行40: ToyotaTundra Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行41: ToyotaSienna Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行42: ToyotaSupra Toyota CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行43: HondaCivic Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行44: HondaAccord Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行45: HondaCRV Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行46: HondaPilot Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行47: HondaOdyssey Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行48: HondaRidgeline Honda CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行49: FordF150 Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行50: FordMustang Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行51: FordExplorer Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行52: FordEscape Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行53: FordRanger Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行54: FordTransit Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行55: FordBronco Ford CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行56: ChevroletSilverado Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行57: ChevroletTahoe Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行58: ChevroletSuburban Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行59: ChevroletCamaro Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行60: ChevroletEquinox Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行61: ChevroletBolt Chevrolet CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行62: TeslaModelS Tesla CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行63: TeslaModel3 Tesla CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行64: TeslaModelX Tesla CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行65: TeslaModelY Tesla CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行66: TeslaCybertruck Tesla CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行67: NissanAltima Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行68: NissanSentra Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行69: NissanRogue Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行70: NissanPathfinder Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行71: NissanLeaf Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行72: NissanFrontier Nissan CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行73: VolkswagenGolf Volkswagen CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行74: VolkswagenJetta Volkswagen CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行75: VolkswagenPassat Volkswagen CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行76: VolkswagenTiguan Volkswagen CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)
- 行77: VolkswagenID4 Volkswagen CarManufacturer  | 原因：Model is not a subclass of manufacturer (brand->manufacturer chain is not inheritance)

## 删除项：Countries/data.txt
- 行4: AmericanSamoa Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行7: Anguilla Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行8: Antarctica Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行12: Aruba Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行24: Bermuda Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行29: BouvetIsland Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行31: BritishIndianOceanTerritory Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行40: CaymanIslands Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行45: ChristmasIsland Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行46: CocosKeelingIslands Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行51: CookIslands Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行70: FalklandIslandsMalvinas Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行71: FaroeIslands Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行75: FrenchGuiana Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行76: FrenchPolynesia Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行77: FrenchSouthernTerritories Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行83: Gibraltar Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行85: Greenland Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行87: Guadeloupe Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行88: Guam Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行90: Guernsey Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行95: HeardIslandandMcDonaldIslands Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行109: Jersey Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行134: Martinique Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country
- 行137: Mayotte Country GeopoliticalEntity  | 原因：Not a sovereign country / widely recognized country (territory/region), so a is not a subclass of Country

## 删除项：Place/data.txt
- 行18: AustraliaContinent Continent GeographicFeature  | 原因：Synthetic term; 'AustraliaContinent' is not a commonly used proper noun class

## 新增项（追加到各自 data.txt 文件末尾）
### Car/data.txt 新增（43条）
- ToyotaCorolla CompactSedan Sedan
- ToyotaPrius HybridCar Car
- ToyotaRAV4 SUV Car
- ToyotaHighlander MidsizeSUV SUV
- ToyotaTacoma MidSizePickup PickupTruck
- ToyotaTundra FullSizePickup PickupTruck
- ToyotaSienna Minivan Van
- ToyotaSupra SportsCar Car
- HondaCivic CompactCar Car
- HondaAccord MidsizeSedan Sedan
- HondaCRV CompactSUV SUV
- HondaPilot FullSizeSUV SUV
- HondaOdyssey PassengerMinivan Minivan
- HondaRidgeline CrewCabPickup PickupTruck
- FordF150 PickupTruck Truck
- FordMustang MuscleCar SportsCar
- FordExplorer FullSizeSUV Vehicle
- FordEscape Crossover Car
- FordRanger LightTruck Truck
- FordTransit CargoVan Van
- FordBronco OffRoadSUV SUV
- ChevroletSilverado FullSizePickup Truck
- ChevroletTahoe SUV Vehicle
- ChevroletSuburban ExtendedSUV SUV
- ChevroletCamaro Coupe Car
- ChevroletEquinox CompactCrossover Crossover
- ChevroletBolt ElectricCar Vehicle
- TeslaModelS ElectricSedan Sedan
- TeslaModel3 Sedan Car
- TeslaModelX ElectricSUV SUV
- TeslaModelY ElectricCrossover Vehicle
- TeslaCybertruck ElectricPickup Truck
- NissanAltima FullSizeSedan Sedan
- NissanSentra EconomySedan Sedan
- NissanRogue CompactSUV Vehicle
- NissanPathfinder ThreeRowSUV SUV
- NissanLeaf ElectricHatchback Hatchback
- NissanFrontier CompactPickup PickupTruck
- VolkswagenGolf Hatchback Car
- VolkswagenJetta Sedan Automobile
- VolkswagenPassat MidsizeCar Automobile
- VolkswagenTiguan CrossoverSUV SUV
- VolkswagenID4 ElectricCrossover Car

### Countries/data.txt 新增（25条）
- Netherlands Kingdom State
- NewZealand IslandNation State
- Nigeria FederalRepublic Republic
- Norway ConstitutionalMonarchy Monarchy
- Pakistan IslamicRepublic Republic
- Peru Republic State
- Philippines ArchipelagoNation State
- Poland ParliamentaryRepublic Republic
- Portugal Republic Country
- Qatar Emirate State
- Romania SemiPresidentialRepublic Republic
- Russia Federation State
- Rwanda PresidentialRepublic Republic
- SaudiArabia AbsoluteMonarchy Monarchy
- Senegal UnitaryRepublic Republic
- Singapore CityState State
- Slovakia CentralEuropeanCountry EuropeanCountry
- Slovenia AlpineCountry EuropeanCountry
- SouthAfrica AfricanCountry Country
- Spain ParliamentaryMonarchy Monarchy
- Sweden Kingdom Country
- Switzerland Confederation State
- Thailand SoutheastAsianCountry AsianCountry
- Tunisia NorthAfricanCountry AfricanCountry
- Turkey TranscontinentalCountry Country

### Place/data.txt 新增（1条）
- Oceania Continent GeographicFeature
