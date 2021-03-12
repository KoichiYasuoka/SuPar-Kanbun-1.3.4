#! /usr/bin/python3 -i
# coding=utf-8
simplify={
  "𡑍":"𫭼",
  "㑳":"㑇",
  "㑹":"会",
  "㘚":"㘎",
  "㠣":"𫵷",
  "㥮":"㤘",
  "㧞":"拔",
  "㨗":"捷",
  "㩁":"榷",
  "㩳":"㧐",
  "㫖":"旨",
  "㮚":"栗",
  "㵎":"涧",
  "䃮":"鿎",
  "䊷":"䌶",
  "䓣":"𬜯",
  "䘏":"恤",
  "䘮":"丧",
  "䟽":"疏",
  "䡵":"𫟦",
  "䥑":"鿏",
  "䥕":"𬭯",
  "䧟":"陷",
  "䮄":"𫠊",
  "䰾":"鲃",
  "䲁":"鳚",
  "䲘":"鳤",
  "䴉":"鹮",
  "両":"两",
  "並":"并",
  "乗":"乘",
  "亀":"龟",
  "亂":"乱",
  "亙":"亘",
  "亜":"亚",
  "亞":"亚",
  "仏":"佛",
  "伝":"传",
  "佔":"占",
  "併":"并",
  "來":"来",
  "侖":"仑",
  "価":"价",
  "侶":"侣",
  "係":"系",
  "俔":"伣",
  "俠":"侠",
  "倀":"伥",
  "倂":"并",
  "倆":"俩",
  "倈":"俫",
  "倉":"仓",
  "個":"个",
  "們":"们",
  "倖":"幸",
  "倣":"仿",
  "倫":"伦",
  "倶":"俱",
  "倹":"俭",
  "偉":"伟",
  "側":"侧",
  "偵":"侦",
  "偸":"偷",
  "偽":"伪",
  "傑":"杰",
  "傖":"伧",
  "傘":"伞",
  "備":"备",
  "傚":"效",
  "傢":"家",
  "傭":"佣",
  "傳":"传",
  "傴":"伛",
  "債":"债",
  "傷":"伤",
  "傾":"倾",
  "僂":"偻",
  "僅":"仅",
  "僉":"佥",
  "僊":"仙",
  "僑":"侨",
  "僕":"仆",
  "僞":"伪",
  "僤":"𫢸",
  "僥":"侥",
  "僨":"偾",
  "僮":"童",
  "價":"价",
  "儀":"仪",
  "儂":"侬",
  "億":"亿",
  "儈":"侩",
  "儉":"俭",
  "儐":"傧",
  "儔":"俦",
  "儕":"侪",
  "儘":"尽",
  "償":"偿",
  "優":"优",
  "儲":"储",
  "儷":"俪",
  "儺":"傩",
  "儻":"傥",
  "儼":"俨",
  "兇":"凶",
  "兌":"兑",
  "兎":"兔",
  "児":"儿",
  "兒":"儿",
  "兗":"兖",
  "內":"内",
  "兩":"两",
  "円":"园",
  "冑":"胄",
  "凍":"冻",
  "処":"处",
  "凱":"凯",
  "別":"别",
  "刦":"劫",
  "刪":"删",
  "刹":"剎",
  "剄":"刭",
  "則":"则",
  "剋":"克",
  "剗":"刬",
  "剛":"刚",
  "剝":"剥",
  "剣":"剑",
  "剮":"剐",
  "剴":"剀",
  "創":"创",
  "劃":"划",
  "劇":"剧",
  "劉":"刘",
  "劊":"刽",
  "劌":"刿",
  "劍":"剑",
  "劑":"剂",
  "劒":"剑",
  "労":"劳",
  "効":"效",
  "勁":"劲",
  "勅":"敕",
  "勑":"敕",
  "動":"动",
  "勗":"勖",
  "務":"务",
  "勛":"勋",
  "勝":"胜",
  "勞":"劳",
  "勢":"势",
  "勣":"𪟝",
  "勦":"剿",
  "勩":"勚",
  "勱":"劢",
  "勲":"勋",
  "勳":"勋",
  "勵":"励",
  "勸":"劝",
  "匭":"匦",
  "匯":"汇",
  "匱":"匮",
  "區":"区",
  "協":"协",
  "単":"单",
  "卹":"恤",
  "卻":"却",
  "卽":"即",
  "厙":"厍",
  "厠":"厕",
  "厤":"历",
  "厭":"厌",
  "厲":"厉",
  "厳":"严",
  "厴":"厣",
  "參":"参",
  "収":"攸",
  "叡":"睿",
  "叢":"丛",
  "吳":"吴",
  "吶":"呐",
  "吿":"告",
  "呂":"吕",
  "呉":"吴",
  "呪":"咒",
  "咼":"呙",
  "員":"员",
  "唄":"呗",
  "問":"问",
  "啓":"启",
  "啗":"啖",
  "啞":"哑",
  "啟":"启",
  "喩":"喻",
  "喪":"丧",
  "喫":"吃",
  "喬":"乔",
  "單":"单",
  "喲":"哟",
  "営":"营",
  "嗆":"呛",
  "嗇":"啬",
  "嗊":"唝",
  "嗎":"吗",
  "嗚":"呜",
  "嗩":"唢",
  "嗶":"哔",
  "嘆":"叹",
  "嘍":"喽",
  "嘑":"呼",
  "嘔":"呕",
  "嘖":"啧",
  "嘗":"尝",
  "嘜":"唛",
  "嘩":"哗",
  "嘮":"唠",
  "嘯":"啸",
  "嘰":"叽",
  "嘵":"哓",
  "嘸":"呒",
  "嘽":"啴",
  "噁":"𫫇",
  "噉":"啖",
  "噝":"咝",
  "噠":"哒",
  "噥":"哝",
  "噦":"哕",
  "噯":"嗳",
  "噲":"哙",
  "噴":"喷",
  "噸":"吨",
  "噹":"当",
  "嚀":"咛",
  "嚇":"吓",
  "嚌":"哜",
  "嚕":"噜",
  "嚙":"啮",
  "嚢":"囊",
  "嚦":"呖",
  "嚨":"咙",
  "嚮":"向",
  "嚲":"亸",
  "嚳":"喾",
  "嚴":"严",
  "嚶":"嘤",
  "囀":"啭",
  "囁":"嗫",
  "囂":"嚣",
  "囅":"冁",
  "囈":"呓",
  "囉":"啰",
  "囌":"苏",
  "囑":"嘱",
  "囲":"围",
  "図":"图",
  "圇":"囵",
  "國":"国",
  "圍":"围",
  "園":"园",
  "圓":"圆",
  "圖":"图",
  "團":"团",
  "垻":"坝",
  "埀":"垂",
  "埡":"垭",
  "埨":"𫭢",
  "執":"执",
  "堅":"坚",
  "堊":"垩",
  "堖":"垴",
  "堝":"埚",
  "堯":"尧",
  "報":"报",
  "場":"场",
  "塊":"块",
  "塋":"茔",
  "塏":"垲",
  "塒":"埘",
  "塗":"涂",
  "塚":"冢",
  "塡":"填",
  "塢":"坞",
  "塤":"埙",
  "塩":"盐",
  "塵":"尘",
  "塸":"𫭟",
  "塹":"堑",
  "塿":"𪣻",
  "墊":"垫",
  "墜":"坠",
  "墠":"𫮃",
  "墮":"堕",
  "墳":"坟",
  "墶":"垯",
  "墻":"墙",
  "墾":"垦",
  "壇":"坛",
  "壊":"坏",
  "壋":"垱",
  "壌":"壤",
  "壎":"埙",
  "壓":"压",
  "壘":"垒",
  "壙":"圹",
  "壚":"垆",
  "壞":"坏",
  "壟":"垄",
  "壢":"坜",
  "壩":"坝",
  "壪":"塆",
  "壯":"壮",
  "壱":"壹",
  "壺":"壶",
  "壻":"婿",
  "壼":"壸",
  "壽":"寿",
  "変":"变",
  "夢":"梦",
  "夥":"伙",
  "夾":"夹",
  "奐":"奂",
  "奥":"奧",
  "奨":"奖",
  "奩":"奁",
  "奪":"夺",
  "奬":"奖",
  "奮":"奋",
  "妝":"妆",
  "妬":"妒",
  "姉":"姊",
  "姦":"奸",
  "姪":"侄",
  "娙":"𫰛",
  "娛":"娱",
  "娯":"娱",
  "婁":"娄",
  "婦":"妇",
  "婭":"娅",
  "媧":"娲",
  "媼":"媪",
  "媽":"妈",
  "嫗":"妪",
  "嫵":"妩",
  "嫻":"娴",
  "嫿":"婳",
  "嬀":"妫",
  "嬃":"媭",
  "嬈":"娆",
  "嬋":"婵",
  "嬌":"娇",
  "嬙":"嫱",
  "嬡":"嫒",
  "嬪":"嫔",
  "嬰":"婴",
  "嬸":"婶",
  "孌":"娈",
  "孫":"孙",
  "學":"学",
  "孼":"孽",
  "孿":"孪",
  "実":"实",
  "宮":"宫",
  "寃":"冤",
  "寖":"浸",
  "寘":"置",
  "寛":"宽",
  "寢":"寝",
  "實":"实",
  "寧":"宁",
  "審":"审",
  "寫":"写",
  "寬":"宽",
  "寵":"宠",
  "寶":"宝",
  "対":"对",
  "専":"专",
  "尅":"克",
  "將":"将",
  "專":"专",
  "尋":"寻",
  "對":"对",
  "導":"导",
  "尙":"尚",
  "尭":"尧",
  "尷":"尴",
  "屍":"尸",
  "屓":"屃",
  "屛":"屏",
  "屢":"屡",
  "層":"层",
  "屨":"屦",
  "屬":"属",
  "岡":"冈",
  "峯":"峰",
  "峴":"岘",
  "島":"岛",
  "峽":"峡",
  "崍":"崃",
  "崗":"岗",
  "崧":"嵩",
  "崬":"岽",
  "嵐":"岚",
  "嵽":"𫶇",
  "嶁":"嵝",
  "嶄":"崭",
  "嶇":"岖",
  "嶔":"嵚",
  "嶗":"崂",
  "嶠":"峤",
  "嶢":"峣",
  "嶧":"峄",
  "嶨":"峃",
  "嶮":"崄",
  "嶸":"嵘",
  "嶺":"岭",
  "嶼":"屿",
  "嶽":"岳",
  "巋":"岿",
  "巌":"岩",
  "巒":"峦",
  "巔":"巅",
  "巖":"岩",
  "巘":"𪩘",
  "巣":"巢",
  "巰":"巯",
  "巵":"卮",
  "巻":"卷",
  "帥":"帅",
  "師":"师",
  "帯":"带",
  "帰":"归",
  "帳":"帐",
  "帶":"带",
  "幀":"帧",
  "幃":"帏",
  "幗":"帼",
  "幘":"帻",
  "幟":"帜",
  "幣":"币",
  "幫":"帮",
  "幬":"帱",
  "幷":"并",
  "幹":"干",
  "幾":"几",
  "広":"广",
  "庫":"库",
  "廁":"厕",
  "廃":"废",
  "廄":"厩",
  "廎":"庼",
  "廐":"厩",
  "廚":"厨",
  "廞":"𫷷",
  "廟":"庙",
  "廠":"厂",
  "廡":"庑",
  "廢":"废",
  "廣":"广",
  "廩":"廪",
  "廬":"庐",
  "廳":"厅",
  "廵":"巡",
  "弒":"弑",
  "弔":"吊",
  "張":"张",
  "強":"强",
  "彄":"𫸩",
  "彆":"别",
  "彈":"弹",
  "彊":"强",
  "彌":"弥",
  "彎":"弯",
  "彙":"汇",
  "彠":"彟",
  "彥":"彦",
  "彫":"雕",
  "彲":"彨",
  "徃":"往",
  "後":"后",
  "徑":"径",
  "従":"从",
  "從":"从",
  "徠":"徕",
  "徧":"遍",
  "復":"复",
  "徳":"德",
  "徴":"征",
  "徵":"征",
  "徹":"彻",
  "応":"应",
  "恆":"恒",
  "恥":"耻",
  "恵":"惠",
  "悅":"悦",
  "悪":"恶",
  "悵":"怅",
  "悶":"闷",
  "悽":"凄",
  "惡":"恶",
  "惱":"恼",
  "惲":"恽",
  "惻":"恻",
  "愛":"爱",
  "愜":"惬",
  "愨":"悫",
  "愴":"怆",
  "愷":"恺",
  "愼":"慎",
  "愾":"忾",
  "慄":"栗",
  "態":"态",
  "慍":"愠",
  "慘":"惨",
  "慙":"惭",
  "慚":"惭",
  "慟":"恸",
  "慣":"惯",
  "慤":"悫",
  "慪":"怄",
  "慫":"怂",
  "慮":"虑",
  "慳":"悭",
  "慴":"慑",
  "慶":"庆",
  "慾":"欲",
  "憂":"忧",
  "憊":"惫",
  "憐":"怜",
  "憑":"凭",
  "憒":"愦",
  "憖":"慭",
  "憚":"惮",
  "憤":"愤",
  "憫":"悯",
  "憮":"怃",
  "憲":"宪",
  "憶":"忆",
  "懇":"恳",
  "應":"应",
  "懌":"怿",
  "懐":"怀",
  "懞":"蒙",
  "懟":"怼",
  "懣":"懑",
  "懨":"恹",
  "懲":"惩",
  "懶":"懒",
  "懷":"怀",
  "懸":"悬",
  "懺":"忏",
  "懼":"惧",
  "懾":"慑",
  "戀":"恋",
  "戇":"戆",
  "戔":"戋",
  "戦":"战",
  "戧":"戗",
  "戩":"戬",
  "戯":"戏",
  "戰":"战",
  "戲":"戏",
  "戶":"户",
  "戸":"户",
  "戹":"厄",
  "戻":"戾",
  "払":"拂",
  "扡":"拖",
  "抜":"拔",
  "択":"择",
  "拏":"拿",
  "拝":"拜",
  "拡":"扩",
  "挙":"举",
  "挾":"挟",
  "挿":"插",
  "捨":"舍",
  "捫":"扪",
  "捲":"卷",
  "掃":"扫",
  "掄":"抡",
  "掛":"挂",
  "採":"采",
  "掲":"揭",
  "揀":"拣",
  "揔":"摠",
  "揚":"扬",
  "換":"换",
  "揮":"挥",
  "搆":"构",
  "損":"损",
  "搖":"摇",
  "搗":"捣",
  "搤":"扼",
  "搶":"抢",
  "摂":"摄",
  "摑":"掴",
  "摜":"掼",
  "摟":"搂",
  "摯":"挚",
  "摳":"抠",
  "摶":"抟",
  "摺":"折",
  "摻":"掺",
  "撃":"击",
  "撅":"噘",
  "撈":"捞",
  "撏":"挦",
  "撓":"挠",
  "撝":"㧑",
  "撣":"掸",
  "撥":"拨",
  "撫":"抚",
  "撲":"扑",
  "撳":"揿",
  "撻":"挞",
  "撾":"挝",
  "撿":"捡",
  "擁":"拥",
  "擄":"掳",
  "擇":"择",
  "擊":"击",
  "擋":"挡",
  "擓":"㧟",
  "擔":"担",
  "據":"据",
  "擠":"挤",
  "擣":"捣",
  "擧":"举",
  "擬":"拟",
  "擯":"摈",
  "擰":"拧",
  "擱":"搁",
  "擲":"掷",
  "擴":"扩",
  "擷":"撷",
  "擺":"摆",
  "擻":"擞",
  "擼":"撸",
  "擾":"扰",
  "攄":"摅",
  "攆":"撵",
  "攏":"拢",
  "攔":"拦",
  "攖":"撄",
  "攙":"搀",
  "攛":"撺",
  "攜":"携",
  "攝":"摄",
  "攢":"攒",
  "攣":"挛",
  "攤":"摊",
  "攪":"搅",
  "攬":"揽",
  "攷":"考",
  "敍":"叙",
  "敎":"教",
  "敗":"败",
  "敘":"叙",
  "敵":"敌",
  "數":"数",
  "敺":"驱",
  "斂":"敛",
  "斃":"毙",
  "斆":"敩",
  "斉":"齐",
  "斎":"斋",
  "斕":"斓",
  "斬":"斩",
  "斲":"斫",
  "斷":"断",
  "於":"于",
  "旂":"旗",
  "旣":"既",
  "昇":"升",
  "時":"时",
  "晉":"晋",
  "晛":"𬀪",
  "晝":"昼",
  "晩":"晚",
  "暁":"晓",
  "暈":"晕",
  "暉":"晖",
  "暐":"𬀩",
  "暘":"旸",
  "暠":"皓",
  "暢":"畅",
  "暦":"历",
  "暫":"暂",
  "暱":"昵",
  "曄":"晔",
  "曆":"历",
  "曇":"昙",
  "曉":"晓",
  "曖":"暧",
  "曠":"旷",
  "曨":"昽",
  "曬":"晒",
  "書":"书",
  "會":"会",
  "朞":"期",
  "朧":"胧",
  "朮":"术",
  "東":"东",
  "柰":"奈",
  "柵":"栅",
  "栄":"荣",
  "桮":"杯",
  "梘":"枧",
  "梜":"𬂩",
  "條":"条",
  "梟":"枭",
  "梲":"棁",
  "棄":"弃",
  "棊":"棋",
  "棖":"枨",
  "棗":"枣",
  "棟":"栋",
  "棡":"㭎",
  "棧":"栈",
  "棲":"栖",
  "棶":"梾",
  "椏":"桠",
  "検":"检",
  "楊":"杨",
  "楓":"枫",
  "楡":"榆",
  "楨":"桢",
  "業":"业",
  "極":"极",
  "楽":"乐",
  "榪":"杩",
  "榮":"荣",
  "榿":"桤",
  "構":"构",
  "槍":"枪",
  "槤":"梿",
  "槧":"椠",
  "槨":"椁",
  "槪":"概",
  "槳":"桨",
  "樁":"桩",
  "樂":"乐",
  "樅":"枞",
  "樓":"楼",
  "標":"标",
  "樞":"枢",
  "樣":"样",
  "権":"权",
  "樸":"朴",
  "樹":"树",
  "樺":"桦",
  "樿":"椫",
  "橈":"桡",
  "橋":"桥",
  "機":"机",
  "橢":"椭",
  "橫":"横",
  "檉":"柽",
  "檔":"档",
  "檜":"桧",
  "檟":"槚",
  "檢":"检",
  "檣":"樯",
  "檮":"梼",
  "檯":"台",
  "檳":"槟",
  "檸":"柠",
  "檻":"槛",
  "櫃":"柜",
  "櫍":"𬃊",
  "櫓":"橹",
  "櫚":"榈",
  "櫛":"栉",
  "櫝":"椟",
  "櫞":"橼",
  "櫟":"栎",
  "櫧":"槠",
  "櫨":"栌",
  "櫪":"枥",
  "櫬":"榇",
  "櫳":"栊",
  "櫸":"榉",
  "櫻":"樱",
  "欄":"栏",
  "權":"权",
  "欏":"椤",
  "欑":"𪴙",
  "欒":"栾",
  "欓":"𣗋",
  "欖":"榄",
  "欞":"棂",
  "欬":"咳",
  "欽":"钦",
  "歎":"叹",
  "歐":"欧",
  "歓":"欢",
  "歛":"敛",
  "歟":"欤",
  "歡":"欢",
  "歩":"步",
  "歯":"齿",
  "歲":"岁",
  "歳":"岁",
  "歴":"历",
  "歷":"历",
  "歸":"归",
  "歿":"没",
  "殀":"夭",
  "殁":"没",
  "殘":"残",
  "殞":"殒",
  "殤":"殇",
  "殫":"殚",
  "殮":"殓",
  "殯":"殡",
  "殰":"㱩",
  "殲":"歼",
  "殺":"杀",
  "殻":"壳",
  "殽":"淆",
  "毀":"毁",
  "毆":"殴",
  "毎":"每",
  "毿":"毵",
  "氂":"牦",
  "氈":"毡",
  "氌":"氇",
  "気":"气",
  "氣":"气",
  "氫":"氢",
  "氬":"氩",
  "氷":"冰",
  "氾":"泛",
  "汎":"汛",
  "汙":"污",
  "汚":"污",
  "決":"决",
  "沒":"没",
  "沖":"冲",
  "沢":"泽",
  "況":"况",
  "泝":"溯",
  "洩":"泄",
  "洶":"汹",
  "浜":"滨",
  "浹":"浃",
  "浿":"𬇙",
  "涇":"泾",
  "涖":"莅",
  "涗":"涚",
  "涙":"泪",
  "涜":"渎",
  "涼":"凉",
  "淚":"泪",
  "淨":"净",
  "淩":"凌",
  "淪":"沦",
  "淵":"渊",
  "淶":"涞",
  "淸":"清",
  "淺":"浅",
  "渇":"渴",
  "済":"济",
  "渉":"涉",
  "渓":"溪",
  "渙":"涣",
  "減":"减",
  "渢":"沨",
  "渦":"涡",
  "測":"测",
  "渾":"浑",
  "湊":"凑",
  "湋":"𣲗",
  "湞":"浈",
  "湧":"涌",
  "湯":"汤",
  "満":"满",
  "準":"准",
  "溝":"沟",
  "溫":"温",
  "溮":"浉",
  "溳":"涢",
  "溼":"湿",
  "滄":"沧",
  "滅":"灭",
  "滌":"涤",
  "滎":"荥",
  "滬":"沪",
  "滯":"滞",
  "滲":"渗",
  "滷":"卤",
  "滸":"浒",
  "滻":"浐",
  "滿":"满",
  "漁":"渔",
  "漊":"溇",
  "漍":"𬇹",
  "漑":"溉",
  "漚":"沤",
  "漢":"汉",
  "漣":"涟",
  "漬":"渍",
  "漲":"涨",
  "漸":"渐",
  "漿":"浆",
  "潁":"颍",
  "潅":"灌",
  "潑":"泼",
  "潔":"洁",
  "潕":"𣲘",
  "潙":"沩",
  "潛":"潜",
  "潜":"潛",
  "潤":"润",
  "潯":"浔",
  "潰":"溃",
  "潷":"滗",
  "潿":"涠",
  "澀":"涩",
  "澆":"浇",
  "澇":"涝",
  "澐":"沄",
  "澗":"涧",
  "澠":"渑",
  "澣":"浣",
  "澤":"泽",
  "澦":"滪",
  "澫":"𬇕",
  "澮":"浍",
  "澱":"淀",
  "濁":"浊",
  "濃":"浓",
  "濆":"𣸣",
  "濕":"湿",
  "濘":"泞",
  "濚":"溁",
  "濛":"蒙",
  "濜":"浕",
  "濟":"济",
  "濤":"涛",
  "濫":"滥",
  "濬":"浚",
  "濰":"潍",
  "濱":"滨",
  "濺":"溅",
  "濼":"泺",
  "濾":"滤",
  "瀂":"澛",
  "瀅":"滢",
  "瀆":"渎",
  "瀉":"泻",
  "瀋":"沈",
  "瀏":"浏",
  "瀕":"濒",
  "瀘":"泸",
  "瀝":"沥",
  "瀟":"潇",
  "瀠":"潆",
  "瀧":"泷",
  "瀨":"濑",
  "瀰":"弥",
  "瀲":"潋",
  "瀾":"澜",
  "灃":"沣",
  "灄":"滠",
  "灑":"洒",
  "灕":"漓",
  "灘":"滩",
  "灝":"灏",
  "灣":"湾",
  "灤":"滦",
  "灧":"滟",
  "災":"灾",
  "為":"为",
  "烏":"乌",
  "烴":"烃",
  "無":"无",
  "焼":"烧",
  "煇":"辉",
  "煉":"炼",
  "煒":"炜",
  "煕":"熙",
  "煙":"烟",
  "煢":"茕",
  "煥":"焕",
  "煩":"烦",
  "煬":"炀",
  "熅":"煴",
  "熒":"荧",
  "熗":"炝",
  "熰":"𬉼",
  "熱":"热",
  "熲":"颎",
  "熾":"炽",
  "燀":"𬊤",
  "燁":"烨",
  "燈":"灯",
  "燒":"烧",
  "燖":"𬊈",
  "燙":"烫",
  "燜":"焖",
  "營":"营",
  "燦":"灿",
  "燭":"烛",
  "燴":"烩",
  "燼":"烬",
  "燾":"焘",
  "爇":"𦶟",
  "爍":"烁",
  "爐":"炉",
  "爛":"烂",
  "爭":"争",
  "爲":"为",
  "爺":"爷",
  "爾":"尔",
  "牀":"床",
  "牆":"墙",
  "牋":"笺",
  "牕":"窗",
  "牘":"牍",
  "牽":"牵",
  "犖":"荦",
  "犠":"牺",
  "犢":"犊",
  "犧":"牺",
  "狀":"状",
  "狥":"徇",
  "狹":"狭",
  "狽":"狈",
  "猟":"猎",
  "猶":"犹",
  "猻":"狲",
  "獁":"犸",
  "獄":"狱",
  "獅":"狮",
  "獎":"奖",
  "獣":"兽",
  "獧":"狷",
  "獨":"独",
  "獪":"狯",
  "獫":"猃",
  "獮":"狝",
  "獰":"狞",
  "獲":"获",
  "獵":"猎",
  "獷":"犷",
  "獸":"兽",
  "獺":"獭",
  "獻":"献",
  "獼":"猕",
  "玀":"猡",
  "玆":"兹",
  "現":"现",
  "琖":"盏",
  "琿":"珲",
  "瑇":"玳",
  "瑋":"玮",
  "瑒":"玚",
  "瑣":"琐",
  "瑤":"瑶",
  "瑩":"莹",
  "瑪":"玛",
  "瑯":"琅",
  "瑲":"玱",
  "璉":"琏",
  "璊":"𫞩",
  "璕":"𬍤",
  "璗":"𬍡",
  "璡":"琎",
  "璣":"玑",
  "璦":"瑷",
  "璫":"珰",
  "環":"环",
  "璵":"玙",
  "璽":"玺",
  "瓅":"𬍛",
  "瓊":"琼",
  "瓏":"珑",
  "瓔":"璎",
  "瓚":"瓒",
  "瓛":"𤩽",
  "甌":"瓯",
  "甕":"瓮",
  "產":"产",
  "産":"产",
  "甯":"宁",
  "畝":"亩",
  "畢":"毕",
  "畧":"略",
  "畫":"画",
  "異":"异",
  "畱":"留",
  "當":"当",
  "疇":"畴",
  "疊":"迭",
  "疎":"疏",
  "痙":"痉",
  "痾":"疴",
  "瘉":"愈",
  "瘋":"疯",
  "瘍":"疡",
  "瘖":"喑",
  "瘞":"瘗",
  "瘡":"疮",
  "瘧":"疟",
  "瘮":"瘆",
  "瘲":"疭",
  "瘻":"瘘",
  "療":"疗",
  "癆":"痨",
  "癇":"痫",
  "癉":"瘅",
  "癘":"疠",
  "癟":"瘪",
  "癢":"痒",
  "癤":"疖",
  "癥":"症",
  "癧":"疬",
  "癩":"癞",
  "癬":"癣",
  "癭":"瘿",
  "癮":"瘾",
  "癰":"痈",
  "癱":"瘫",
  "癲":"癫",
  "発":"发",
  "發":"发",
  "皁":"皂",
  "皐":"皋",
  "皚":"皑",
  "皜":"皓",
  "皸":"皲",
  "皺":"皱",
  "盜":"盗",
  "盞":"盏",
  "盡":"尽",
  "監":"监",
  "盤":"盘",
  "盧":"卢",
  "盪":"荡",
  "県":"县",
  "眞":"真",
  "眥":"眦",
  "眾":"众",
  "睍":"𪾢",
  "睏":"困",
  "睞":"睐",
  "瞘":"眍",
  "瞜":"䁖",
  "瞞":"瞒",
  "瞼":"睑",
  "矇":"蒙",
  "矓":"眬",
  "矙":"瞰",
  "矚":"瞩",
  "矯":"矫",
  "硃":"朱",
  "硜":"硁",
  "硤":"硖",
  "硨":"砗",
  "硯":"砚",
  "碩":"硕",
  "碭":"砀",
  "碸":"砜",
  "確":"确",
  "碼":"码",
  "磑":"硙",
  "磚":"砖",
  "磣":"碜",
  "磧":"碛",
  "磯":"矶",
  "磽":"硗",
  "磾":"䃅",
  "礄":"硚",
  "礎":"础",
  "礐":"𬒈",
  "礙":"碍",
  "礦":"矿",
  "礪":"砺",
  "礫":"砾",
  "礬":"矾",
  "礱":"砻",
  "祕":"秘",
  "祿":"禄",
  "禍":"祸",
  "禎":"祯",
  "禕":"祎",
  "禡":"祃",
  "禦":"御",
  "禪":"禅",
  "禮":"礼",
  "禰":"祢",
  "禱":"祷",
  "禿":"秃",
  "秊":"年",
  "秖":"只",
  "稅":"税",
  "稟":"禀",
  "種":"种",
  "稱":"称",
  "稲":"稻",
  "稾":"稿",
  "穀":"谷",
  "穇":"䅟",
  "穌":"稣",
  "積":"积",
  "穎":"颖",
  "穠":"秾",
  "穡":"穑",
  "穢":"秽",
  "穣":"穰",
  "穩":"稳",
  "穫":"获",
  "窓":"窗",
  "窩":"窝",
  "窪":"洼",
  "窮":"穷",
  "窵":"窎",
  "窶":"窭",
  "窺":"窥",
  "竄":"窜",
  "竅":"窍",
  "竇":"窦",
  "竈":"灶",
  "竊":"窃",
  "竜":"龙",
  "竝":"并",
  "竪":"竖",
  "競":"竞",
  "筆":"笔",
  "筧":"笕",
  "筩":"筒",
  "筯":"箸",
  "筴":"䇲",
  "箇":"个",
  "箋":"笺",
  "箠":"棰",
  "節":"节",
  "範":"范",
  "築":"筑",
  "篋":"箧",
  "篔":"筼",
  "篢":"𬕂",
  "篤":"笃",
  "篩":"筛",
  "篳":"筚",
  "篹":"纂",
  "簀":"箦",
  "簍":"篓",
  "簒":"篡",
  "簞":"箪",
  "簡":"简",
  "簣":"篑",
  "簫":"箫",
  "簹":"筜",
  "簽":"签",
  "簾":"帘",
  "籃":"篮",
  "籌":"筹",
  "籙":"箓",
  "籛":"篯",
  "籜":"箨",
  "籟":"籁",
  "籠":"笼",
  "籤":"签",
  "籩":"笾",
  "籪":"簖",
  "籬":"篱",
  "籮":"箩",
  "籲":"吁",
  "粛":"肃",
  "粵":"粤",
  "糝":"糁",
  "糞":"粪",
  "糧":"粮",
  "糰":"团",
  "糲":"粝",
  "糴":"籴",
  "糶":"粜",
  "糺":"纠",
  "糾":"纠",
  "紀":"纪",
  "紂":"纣",
  "紃":"𬘓",
  "約":"约",
  "紅":"红",
  "紆":"纡",
  "紇":"纥",
  "紈":"纨",
  "紉":"纫",
  "紋":"纹",
  "納":"纳",
  "紐":"纽",
  "紓":"纾",
  "純":"纯",
  "紕":"纰",
  "紖":"纼",
  "紗":"纱",
  "紘":"纮",
  "紙":"纸",
  "級":"级",
  "紛":"纷",
  "紜":"纭",
  "紝":"纴",
  "紞":"𬘘",
  "紟":"𫄛",
  "紡":"纺",
  "紬":"䌷",
  "細":"细",
  "紱":"绂",
  "紲":"绁",
  "紳":"绅",
  "紵":"纻",
  "紹":"绍",
  "紺":"绀",
  "紼":"绋",
  "紿":"绐",
  "絀":"绌",
  "絁":"𫄟",
  "終":"终",
  "絃":"弦",
  "組":"组",
  "絅":"䌹",
  "絆":"绊",
  "経":"经",
  "絎":"绗",
  "絏":"绁",
  "結":"结",
  "絕":"绝",
  "絝":"绔",
  "絞":"绞",
  "絡":"络",
  "絢":"绚",
  "給":"给",
  "絨":"绒",
  "絪":"𬘡",
  "絰":"绖",
  "統":"统",
  "絲":"丝",
  "絳":"绛",
  "絵":"绘",
  "絶":"绝",
  "絹":"绢",
  "絺":"𫄨",
  "綁":"绑",
  "綃":"绡",
  "綄":"𬘫",
  "綆":"绠",
  "綈":"绨",
  "綉":"绣",
  "綌":"绤",
  "綎":"𬘩",
  "綏":"绥",
  "經":"经",
  "綖":"𫄧",
  "継":"继",
  "続":"续",
  "綜":"综",
  "綝":"𬘭",
  "綠":"绿",
  "綡":"𫟅",
  "綢":"绸",
  "綣":"绻",
  "綧":"𬘯",
  "綪":"𬘬",
  "綫":"线",
  "綬":"绶",
  "維":"维",
  "綯":"绹",
  "綰":"绾",
  "綱":"纲",
  "網":"网",
  "綳":"绷",
  "綴":"缀",
  "綵":"彩",
  "綸":"纶",
  "綹":"绺",
  "綺":"绮",
  "綻":"绽",
  "綽":"绰",
  "綾":"绫",
  "綿":"绵",
  "緄":"绲",
  "緇":"缁",
  "緊":"紧",
  "緋":"绯",
  "緑":"绿",
  "緒":"绪",
  "緔":"绱",
  "緖":"绪",
  "緗":"缃",
  "緘":"缄",
  "緙":"缂",
  "線":"缐",
  "緜":"绵",
  "緝":"缉",
  "緞":"缎",
  "締":"缔",
  "緡":"缗",
  "緣":"缘",
  "緤":"𫄬",
  "緦":"缌",
  "編":"编",
  "緩":"缓",
  "緬":"缅",
  "緯":"纬",
  "緱":"缑",
  "緲":"缈",
  "練":"练",
  "緹":"缇",
  "緻":"致",
  "緼":"缊",
  "縁":"缘",
  "縄":"绳",
  "縈":"萦",
  "縉":"缙",
  "縊":"缢",
  "縋":"缒",
  "縐":"绉",
  "縑":"缣",
  "縕":"缊",
  "縗":"缞",
  "縛":"缚",
  "縝":"缜",
  "縞":"缟",
  "縟":"缛",
  "縣":"县",
  "縦":"纵",
  "縧":"绦",
  "縫":"缝",
  "縭":"缡",
  "縮":"缩",
  "縯":"𬙂",
  "縰":"𫄳",
  "縱":"纵",
  "縲":"缧",
  "縴":"纤",
  "縵":"缦",
  "縶":"絷",
  "縷":"缕",
  "縹":"缥",
  "總":"总",
  "績":"绩",
  "繅":"缫",
  "繆":"缪",
  "繊":"纤",
  "繋":"系",
  "繍":"绣",
  "繒":"缯",
  "織":"织",
  "繕":"缮",
  "繚":"缭",
  "繞":"绕",
  "繡":"绣",
  "繢":"缋",
  "繩":"绳",
  "繪":"绘",
  "繫":"系",
  "繭":"茧",
  "繮":"缰",
  "繯":"缳",
  "繰":"缲",
  "繳":"缴",
  "繶":"𫄷",
  "繹":"绎",
  "繻":"𦈡",
  "繼":"继",
  "繽":"缤",
  "繾":"缱",
  "纁":"𫄸",
  "纆":"𬙊",
  "纈":"缬",
  "纊":"纩",
  "續":"续",
  "纍":"累",
  "纏":"缠",
  "纓":"缨",
  "纔":"才",
  "纕":"𬙋",
  "纖":"纤",
  "纘":"缵",
  "纜":"缆",
  "罃":"䓨",
  "罌":"罂",
  "罎":"坛",
  "罰":"罚",
  "罵":"骂",
  "罷":"罢",
  "羅":"罗",
  "羆":"罴",
  "羈":"羁",
  "羋":"芈",
  "羣":"群",
  "羥":"羟",
  "羨":"羡",
  "義":"义",
  "羶":"膻",
  "習":"习",
  "翬":"翚",
  "翹":"翘",
  "翽":"翙",
  "耬":"耧",
  "耮":"耢",
  "聖":"圣",
  "聞":"闻",
  "聡":"聪",
  "聯":"联",
  "聰":"聪",
  "聲":"声",
  "聳":"耸",
  "聴":"听",
  "聵":"聩",
  "聶":"聂",
  "職":"职",
  "聹":"聍",
  "聽":"听",
  "聾":"聋",
  "肅":"肃",
  "胷":"胸",
  "脅":"胁",
  "脈":"脉",
  "脛":"胫",
  "脣":"唇",
  "脩":"修",
  "脫":"脱",
  "脹":"胀",
  "腎":"肾",
  "腖":"胨",
  "腡":"脶",
  "腦":"脑",
  "腫":"肿",
  "腸":"肠",
  "膕":"腘",
  "膚":"肤",
  "膞":"䏝",
  "膠":"胶",
  "膢":"𦝼",
  "膩":"腻",
  "膽":"胆",
  "膾":"脍",
  "膿":"脓",
  "臉":"脸",
  "臍":"脐",
  "臏":"膑",
  "臘":"腊",
  "臚":"胪",
  "臝":"裸",
  "臟":"脏",
  "臠":"脔",
  "臢":"臜",
  "臥":"卧",
  "臨":"临",
  "臺":"台",
  "與":"与",
  "興":"兴",
  "舉":"举",
  "舊":"旧",
  "舎":"舍",
  "舖":"铺",
  "艙":"舱",
  "艤":"舣",
  "艦":"舰",
  "艫":"舻",
  "艱":"艰",
  "艷":"艳",
  "芻":"刍",
  "茍":"苟",
  "茲":"兹",
  "荅":"答",
  "荊":"荆",
  "荘":"庄",
  "莊":"庄",
  "莖":"茎",
  "莢":"荚",
  "莧":"苋",
  "華":"华",
  "萇":"苌",
  "萊":"莱",
  "萬":"万",
  "萵":"莴",
  "葉":"叶",
  "葒":"荭",
  "葤":"荮",
  "葦":"苇",
  "葷":"荤",
  "蒐":"搜",
  "蒓":"莼",
  "蒔":"莳",
  "蒞":"莅",
  "蒼":"苍",
  "蓀":"荪",
  "蓋":"盖",
  "蓧":"𦰏",
  "蓮":"莲",
  "蓯":"苁",
  "蓽":"荜",
  "蔄":"𬜬",
  "蔔":"卜",
  "蔞":"蒌",
  "蔣":"蒋",
  "蔦":"茑",
  "蔭":"荫",
  "蔵":"藏",
  "蔿":"𫇭",
  "蕁":"荨",
  "蕆":"蒇",
  "蕎":"荞",
  "蕒":"荬",
  "蕓":"芸",
  "蕕":"莸",
  "蕘":"荛",
  "蕚":"萼",
  "蕝":"𫈵",
  "蕢":"蒉",
  "蕩":"荡",
  "蕪":"芜",
  "蕭":"萧",
  "蕷":"蓣",
  "薈":"荟",
  "薊":"蓟",
  "薌":"芗",
  "薑":"姜",
  "薔":"蔷",
  "薘":"荙",
  "薟":"莶",
  "薦":"荐",
  "薩":"萨",
  "薫":"薰",
  "薬":"药",
  "薴":"苧",
  "薺":"荠",
  "藍":"蓝",
  "藎":"荩",
  "藝":"艺",
  "藥":"药",
  "藪":"薮",
  "藭":"䓖",
  "藴":"蕴",
  "藶":"苈",
  "藹":"蔼",
  "藺":"蔺",
  "蘀":"萚",
  "蘄":"蕲",
  "蘆":"芦",
  "蘇":"苏",
  "蘊":"蕴",
  "蘋":"𬞟",
  "蘚":"藓",
  "蘞":"蔹",
  "蘢":"茏",
  "蘭":"兰",
  "蘺":"蓠",
  "蘿":"萝",
  "虆":"蔂",
  "虉":"𬟁",
  "處":"处",
  "虛":"虚",
  "虜":"虏",
  "號":"号",
  "虧":"亏",
  "蛍":"萤",
  "蛺":"蛱",
  "蜆":"蚬",
  "蜋":"螂",
  "蝀":"𬟽",
  "蝕":"蚀",
  "蝦":"虾",
  "蝸":"蜗",
  "蝿":"蝇",
  "螄":"蛳",
  "螞":"蚂",
  "螢":"萤",
  "螮":"䗖",
  "螻":"蝼",
  "蟄":"蛰",
  "蟈":"蝈",
  "蟎":"螨",
  "蟜":"𫊸",
  "蟣":"虮",
  "蟬":"蝉",
  "蟯":"蛲",
  "蟲":"虫",
  "蟶":"蛏",
  "蟻":"蚁",
  "蠅":"蝇",
  "蠆":"虿",
  "蠐":"蛴",
  "蠑":"蝾",
  "蠟":"蜡",
  "蠣":"蛎",
  "蠨":"蟏",
  "蠭":"蜂",
  "蠱":"蛊",
  "蠶":"蚕",
  "蠻":"蛮",
  "衆":"众",
  "衊":"蔑",
  "術":"术",
  "衛":"卫",
  "衝":"冲",
  "衞":"卫",
  "衹":"只",
  "袞":"衮",
  "裊":"袅",
  "裏":"里",
  "補":"补",
  "裝":"装",
  "裡":"里",
  "製":"制",
  "複":"复",
  "褌":"裈",
  "褘":"袆",
  "褲":"裤",
  "褳":"裢",
  "褸":"褛",
  "褻":"亵",
  "襀":"𫌀",
  "襃":"褒",
  "襇":"裥",
  "襏":"袯",
  "襓":"𫋹",
  "襖":"袄",
  "襝":"裣",
  "襠":"裆",
  "襢":"袒",
  "襤":"褴",
  "襪":"袜",
  "襬":"摆",
  "襯":"衬",
  "襲":"袭",
  "襴":"襕",
  "覇":"霸",
  "覈":"核",
  "覊":"羁",
  "見":"见",
  "覎":"觃",
  "規":"规",
  "覓":"觅",
  "視":"视",
  "覘":"觇",
  "覚":"觉",
  "覡":"觋",
  "覦":"觎",
  "覧":"览",
  "覩":"睹",
  "親":"亲",
  "覬":"觊",
  "覯":"觏",
  "覲":"觐",
  "観":"观",
  "覷":"觑",
  "覺":"觉",
  "覽":"览",
  "覿":"觌",
  "觀":"观",
  "觴":"觞",
  "觶":"觯",
  "觸":"触",
  "訂":"订",
  "訃":"讣",
  "計":"计",
  "訊":"讯",
  "訌":"讧",
  "討":"讨",
  "訏":"𬣙",
  "訐":"讦",
  "訑":"𫍙",
  "訒":"讱",
  "訓":"训",
  "訕":"讪",
  "訖":"讫",
  "託":"托",
  "記":"记",
  "訛":"讹",
  "訝":"讶",
  "訟":"讼",
  "訢":"䜣",
  "訣":"诀",
  "訥":"讷",
  "訩":"讻",
  "訪":"访",
  "設":"设",
  "許":"许",
  "訴":"诉",
  "訶":"诃",
  "診":"诊",
  "証":"证",
  "詁":"诂",
  "詆":"诋",
  "詎":"讵",
  "詐":"诈",
  "詒":"诒",
  "詔":"诏",
  "評":"评",
  "詖":"诐",
  "詗":"诇",
  "詘":"诎",
  "詛":"诅",
  "詝":"𬣞",
  "詞":"词",
  "詠":"咏",
  "詡":"诩",
  "詢":"询",
  "詣":"诣",
  "試":"试",
  "詩":"诗",
  "詪":"𬣳",
  "詫":"诧",
  "詬":"诟",
  "詭":"诡",
  "詮":"诠",
  "詰":"诘",
  "話":"话",
  "該":"该",
  "詳":"详",
  "詵":"诜",
  "詷":"𫍣",
  "詼":"诙",
  "詿":"诖",
  "誄":"诔",
  "誅":"诛",
  "誆":"诓",
  "誇":"夸",
  "誌":"志",
  "認":"认",
  "誑":"诳",
  "誕":"诞",
  "誘":"诱",
  "誚":"诮",
  "語":"语",
  "誠":"诚",
  "誡":"诫",
  "誣":"诬",
  "誤":"误",
  "誥":"诰",
  "誦":"诵",
  "誨":"诲",
  "說":"说",
  "説":"说",
  "読":"读",
  "誰":"谁",
  "課":"课",
  "誶":"谇",
  "誹":"诽",
  "誼":"谊",
  "誾":"訚",
  "調":"调",
  "諂":"谄",
  "諄":"谆",
  "談":"谈",
  "諉":"诿",
  "請":"请",
  "諍":"诤",
  "諏":"诹",
  "諑":"诼",
  "諒":"谅",
  "諓":"𬣡",
  "論":"论",
  "諗":"谂",
  "諛":"谀",
  "諜":"谍",
  "諝":"谞",
  "諞":"谝",
  "諟":"𬤊",
  "諠":"喧",
  "諡":"谥",
  "諢":"诨",
  "諤":"谔",
  "諦":"谛",
  "諧":"谐",
  "諫":"谏",
  "諭":"谕",
  "諮":"咨",
  "諱":"讳",
  "諲":"𬤇",
  "諳":"谙",
  "諴":"𫍯",
  "諶":"谌",
  "諷":"讽",
  "諸":"诸",
  "諺":"谚",
  "諼":"谖",
  "諾":"诺",
  "謀":"谋",
  "謁":"谒",
  "謂":"谓",
  "謄":"誊",
  "謅":"诌",
  "謊":"谎",
  "謎":"谜",
  "謏":"𫍲",
  "謐":"谧",
  "謔":"谑",
  "謖":"谡",
  "謗":"谤",
  "謙":"谦",
  "謚":"谥",
  "講":"讲",
  "謝":"谢",
  "謠":"谣",
  "謡":"谣",
  "謨":"谟",
  "謫":"谪",
  "謬":"谬",
  "謭":"谫",
  "謳":"讴",
  "謹":"谨",
  "謾":"谩",
  "譁":"华",
  "證":"证",
  "譎":"谲",
  "譏":"讥",
  "譓":"𬤝",
  "譔":"撰",
  "譖":"谮",
  "識":"识",
  "譙":"谯",
  "譚":"谭",
  "譜":"谱",
  "譞":"𫍽",
  "譟":"噪",
  "譫":"谵",
  "譯":"译",
  "議":"议",
  "譲":"让",
  "譴":"谴",
  "護":"护",
  "譽":"誉",
  "讀":"读",
  "變":"变",
  "讋":"詟",
  "讎":"雠",
  "讐":"雠",
  "讒":"谗",
  "讓":"让",
  "讕":"谰",
  "讖":"谶",
  "讙":"欢",
  "讚":"赞",
  "讜":"谠",
  "讞":"谳",
  "谿":"溪",
  "豈":"岂",
  "豊":"丰",
  "豎":"竖",
  "豐":"丰",
  "豬":"猪",
  "豶":"豮",
  "貍":"狸",
  "貓":"猫",
  "貙":"䝙",
  "貝":"贝",
  "貞":"贞",
  "負":"负",
  "財":"财",
  "貢":"贡",
  "貧":"贫",
  "貨":"货",
  "販":"贩",
  "貪":"贪",
  "貫":"贯",
  "責":"责",
  "貯":"贮",
  "貰":"贳",
  "貲":"赀",
  "貳":"贰",
  "貴":"贵",
  "貶":"贬",
  "買":"买",
  "貸":"贷",
  "貺":"贶",
  "費":"费",
  "貼":"贴",
  "貽":"贻",
  "貿":"贸",
  "賀":"贺",
  "賁":"贲",
  "賂":"赂",
  "賃":"赁",
  "賄":"贿",
  "賅":"赅",
  "資":"资",
  "賈":"贾",
  "賊":"贼",
  "賑":"赈",
  "賒":"赊",
  "賓":"宾",
  "賕":"赇",
  "賙":"赒",
  "賚":"赉",
  "賛":"赞",
  "賜":"赐",
  "賞":"赏",
  "賠":"赔",
  "賡":"赓",
  "賢":"贤",
  "賣":"卖",
  "賤":"贱",
  "賦":"赋",
  "賧":"赕",
  "質":"质",
  "賫":"赍",
  "賬":"账",
  "賭":"赌",
  "賴":"赖",
  "賵":"赗",
  "賷":"赍",
  "賺":"赚",
  "賻":"赙",
  "購":"购",
  "賽":"赛",
  "賾":"赜",
  "贄":"贽",
  "贅":"赘",
  "贇":"赟",
  "贈":"赠",
  "贊":"赞",
  "贋":"赝",
  "贍":"赡",
  "贏":"赢",
  "贐":"赆",
  "贓":"赃",
  "贔":"赑",
  "贖":"赎",
  "贛":"赣",
  "贜":"赃",
  "赬":"赪",
  "趕":"赶",
  "趙":"赵",
  "趨":"趋",
  "趲":"趱",
  "跡":"迹",
  "踐":"践",
  "踰":"逾",
  "踴":"踊",
  "蹌":"跄",
  "蹕":"跸",
  "蹣":"蹒",
  "蹤":"踪",
  "蹺":"跷",
  "躂":"跶",
  "躉":"趸",
  "躊":"踌",
  "躋":"跻",
  "躍":"跃",
  "躑":"踯",
  "躒":"跞",
  "躓":"踬",
  "躕":"蹰",
  "躚":"跹",
  "躡":"蹑",
  "躥":"蹿",
  "躦":"躜",
  "躪":"躏",
  "軀":"躯",
  "車":"车",
  "軋":"轧",
  "軌":"轨",
  "軍":"军",
  "軏":"𫐄",
  "軑":"轪",
  "軒":"轩",
  "軔":"轫",
  "軛":"轭",
  "軝":"𬨂",
  "軟":"软",
  "転":"转",
  "軤":"轷",
  "軨":"𫐉",
  "軫":"轸",
  "軲":"轱",
  "軸":"轴",
  "軹":"轵",
  "軺":"轺",
  "軻":"轲",
  "軼":"轶",
  "軽":"轻",
  "軾":"轼",
  "較":"较",
  "輄":"𨐈",
  "輅":"辂",
  "輇":"辁",
  "輈":"辀",
  "載":"载",
  "輊":"轾",
  "輋":"𪨶",
  "輒":"辄",
  "輓":"挽",
  "輔":"辅",
  "輕":"轻",
  "輗":"𫐐",
  "輛":"辆",
  "輜":"辎",
  "輝":"辉",
  "輞":"辋",
  "輟":"辍",
  "輥":"辊",
  "輦":"辇",
  "輩":"辈",
  "輪":"轮",
  "輬":"辌",
  "輭":"软",
  "輮":"𫐓",
  "輯":"辑",
  "輳":"辏",
  "輶":"𬨎",
  "輸":"输",
  "輻":"辐",
  "輼":"辒",
  "輾":"辗",
  "輿":"舆",
  "轀":"辒",
  "轂":"毂",
  "轄":"辖",
  "轅":"辕",
  "轆":"辘",
  "轉":"转",
  "轍":"辙",
  "轎":"轿",
  "轔":"辚",
  "轟":"轰",
  "轡":"辔",
  "轢":"轹",
  "轤":"轳",
  "辦":"办",
  "辭":"辞",
  "辮":"辫",
  "辯":"辩",
  "農":"农",
  "辺":"边",
  "迴":"回",
  "逓":"递",
  "逕":"迳",
  "這":"这",
  "連":"连",
  "進":"进",
  "遅":"迟",
  "遊":"游",
  "運":"运",
  "過":"过",
  "達":"达",
  "違":"违",
  "遙":"遥",
  "遜":"逊",
  "遞":"递",
  "遠":"远",
  "遡":"溯",
  "適":"适",
  "遯":"遁",
  "遲":"迟",
  "遷":"迁",
  "選":"选",
  "遺":"遗",
  "遼":"辽",
  "邁":"迈",
  "還":"还",
  "邇":"迩",
  "邊":"边",
  "邏":"逻",
  "邐":"逦",
  "郞":"郎",
  "郟":"郏",
  "郤":"却",
  "郵":"邮",
  "郷":"乡",
  "鄆":"郓",
  "鄉":"乡",
  "鄒":"邹",
  "鄔":"邬",
  "鄕":"乡",
  "鄖":"郧",
  "鄧":"邓",
  "鄩":"𬩽",
  "鄭":"郑",
  "鄰":"邻",
  "鄲":"郸",
  "鄳":"𫑡",
  "鄴":"邺",
  "鄶":"郐",
  "鄺":"邝",
  "酇":"酂",
  "酈":"郦",
  "酔":"醉",
  "酖":"鸩",
  "醆":"盏",
  "醖":"酝",
  "醜":"丑",
  "醞":"酝",
  "醤":"酱",
  "醫":"医",
  "醬":"酱",
  "醱":"酦",
  "醲":"𬪩",
  "釀":"酿",
  "釁":"衅",
  "釃":"酾",
  "釅":"酽",
  "釈":"释",
  "釋":"释",
  "釐":"厘",
  "釓":"钆",
  "釔":"钇",
  "釕":"钌",
  "釗":"钊",
  "釘":"钉",
  "釙":"钋",
  "針":"针",
  "釣":"钓",
  "釤":"钐",
  "釧":"钏",
  "釩":"钒",
  "釴":"𬬩",
  "釵":"钗",
  "釷":"钍",
  "釹":"钕",
  "釺":"钎",
  "釿":"𬬱",
  "鈀":"钯",
  "鈁":"钫",
  "鈃":"钘",
  "鈄":"钭",
  "鈇":"𫓧",
  "鈈":"钚",
  "鈉":"钠",
  "鈍":"钝",
  "鈎":"钩",
  "鈐":"钤",
  "鈑":"钣",
  "鈔":"钞",
  "鈕":"钮",
  "鈞":"钧",
  "鈣":"钙",
  "鈥":"钬",
  "鈦":"钛",
  "鈧":"钪",
  "鈮":"铌",
  "鈰":"铈",
  "鈴":"铃",
  "鈷":"钴",
  "鈸":"钹",
  "鈹":"铍",
  "鈺":"钰",
  "鈾":"铀",
  "鈿":"钿",
  "鉀":"钾",
  "鉄":"铁",
  "鉅":"钜",
  "鉈":"铊",
  "鉉":"铉",
  "鉊":"𬬿",
  "鉍":"铋",
  "鉑":"铂",
  "鉕":"钷",
  "鉗":"钳",
  "鉚":"铆",
  "鉛":"铅",
  "鉞":"钺",
  "鉢":"钵",
  "鉤":"钩",
  "鉥":"𬬸",
  "鉦":"钲",
  "鉧":"𬭁",
  "鉬":"钼",
  "鉭":"钽",
  "鉮":"𬬹",
  "鉶":"铏",
  "鉷":"𫟹",
  "鉸":"铰",
  "鉺":"铒",
  "鉻":"铬",
  "鉿":"铪",
  "銀":"银",
  "銃":"铳",
  "銅":"铜",
  "銈":"𫓯",
  "銍":"铚",
  "銑":"铣",
  "銓":"铨",
  "銖":"铢",
  "銘":"铭",
  "銚":"铫",
  "銜":"衔",
  "銠":"铑",
  "銣":"铷",
  "銥":"铱",
  "銦":"铟",
  "銨":"铵",
  "銩":"铥",
  "銪":"铕",
  "銫":"铯",
  "銬":"铐",
  "銭":"钱",
  "銱":"铞",
  "銳":"锐",
  "銶":"𨱇",
  "銷":"销",
  "銹":"锈",
  "銻":"锑",
  "銼":"锉",
  "鋁":"铝",
  "鋃":"锒",
  "鋅":"锌",
  "鋇":"钡",
  "鋌":"铤",
  "鋏":"铗",
  "鋐":"𬭎",
  "鋒":"锋",
  "鋗":"𫓶",
  "鋙":"铻",
  "鋝":"锊",
  "鋟":"锓",
  "鋣":"铘",
  "鋤":"锄",
  "鋥":"锃",
  "鋦":"锔",
  "鋨":"锇",
  "鋪":"铺",
  "鋭":"锐",
  "鋮":"铖",
  "鋯":"锆",
  "鋰":"锂",
  "鋱":"铽",
  "鋳":"铸",
  "鋶":"锍",
  "鋸":"锯",
  "鋹":"𬬮",
  "鋼":"钢",
  "錀":"𬬭",
  "錁":"锞",
  "錄":"录",
  "錆":"锖",
  "錇":"锫",
  "錈":"锩",
  "錐":"锥",
  "錒":"锕",
  "錕":"锟",
  "錘":"锤",
  "錙":"锱",
  "錚":"铮",
  "錛":"锛",
  "錞":"𬭚",
  "錟":"锬",
  "錠":"锭",
  "錡":"锜",
  "錢":"钱",
  "錤":"𫓹",
  "錦":"锦",
  "錨":"锚",
  "錫":"锡",
  "錬":"炼",
  "錮":"锢",
  "錯":"错",
  "録":"录",
  "錳":"锰",
  "錶":"表",
  "錸":"铼",
  "鍀":"锝",
  "鍁":"锨",
  "鍃":"锪",
  "鍆":"钔",
  "鍇":"锴",
  "鍈":"锳",
  "鍊":"炼",
  "鍋":"锅",
  "鍍":"镀",
  "鍔":"锷",
  "鍘":"铡",
  "鍚":"钖",
  "鍛":"锻",
  "鍠":"锽",
  "鍤":"锸",
  "鍥":"锲",
  "鍩":"锘",
  "鍬":"锹",
  "鍭":"𬭤",
  "鍰":"锾",
  "鍵":"键",
  "鍶":"锶",
  "鍺":"锗",
  "鍾":"锺",
  "鎂":"镁",
  "鎄":"锿",
  "鎇":"镅",
  "鎊":"镑",
  "鎓":"𬭩",
  "鎔":"镕",
  "鎖":"锁",
  "鎗":"枪",
  "鎘":"镉",
  "鎛":"镈",
  "鎝":"𨱏",
  "鎡":"镃",
  "鎢":"钨",
  "鎣":"蓥",
  "鎦":"镏",
  "鎧":"铠",
  "鎩":"铩",
  "鎪":"锼",
  "鎬":"镐",
  "鎭":"镇",
  "鎮":"镇",
  "鎰":"镒",
  "鎳":"镍",
  "鎵":"镓",
  "鎸":"镌",
  "鎿":"镎",
  "鏃":"镞",
  "鏇":"旋",
  "鏈":"链",
  "鏌":"镆",
  "鏏":"𬭬",
  "鏐":"镠",
  "鏑":"镝",
  "鏗":"铿",
  "鏘":"锵",
  "鏜":"镗",
  "鏝":"镘",
  "鏞":"镛",
  "鏟":"铲",
  "鏡":"镜",
  "鏢":"镖",
  "鏤":"镂",
  "鏨":"錾",
  "鏰":"镚",
  "鏵":"铧",
  "鏷":"镤",
  "鏹":"镪",
  "鏺":"䥽",
  "鏻":"𬭸",
  "鐃":"铙",
  "鐄":"𨱑",
  "鐇":"𫔍",
  "鐋":"铴",
  "鐍":"𫔎",
  "鐏":"𨱔",
  "鐐":"镣",
  "鐒":"铹",
  "鐓":"镦",
  "鐔":"镡",
  "鐘":"钟",
  "鐙":"镫",
  "鐝":"镢",
  "鐠":"镨",
  "鐦":"锎",
  "鐧":"锏",
  "鐨":"镄",
  "鐩":"𬭼",
  "鐮":"镰",
  "鐯":"䦃",
  "鐲":"镯",
  "鐳":"镭",
  "鐵":"铁",
  "鐶":"镮",
  "鐸":"铎",
  "鐺":"铛",
  "鐽":"𫟼",
  "鐿":"镱",
  "鑄":"铸",
  "鑊":"镬",
  "鑌":"镔",
  "鑑":"鉴",
  "鑒":"鉴",
  "鑔":"镲",
  "鑕":"锧",
  "鑞":"镴",
  "鑠":"铄",
  "鑣":"镳",
  "鑥":"镥",
  "鑪":"𬬻",
  "鑭":"镧",
  "鑰":"钥",
  "鑱":"镵",
  "鑲":"镶",
  "鑷":"镊",
  "鑹":"镩",
  "鑼":"锣",
  "鑽":"钻",
  "鑾":"銮",
  "鑿":"凿",
  "钂":"镋",
  "長":"长",
  "門":"门",
  "閂":"闩",
  "閃":"闪",
  "閆":"闫",
  "閈":"闬",
  "閉":"闭",
  "開":"开",
  "閌":"闶",
  "閎":"闳",
  "閏":"闰",
  "閑":"闲",
  "閒":"间",
  "間":"间",
  "閔":"闵",
  "閘":"闸",
  "閡":"阂",
  "関":"关",
  "閣":"阁",
  "閤":"合",
  "閥":"阀",
  "閨":"闺",
  "閩":"闽",
  "閫":"阃",
  "閬":"阆",
  "閭":"闾",
  "閱":"阅",
  "閲":"阅",
  "閶":"阊",
  "閹":"阉",
  "閻":"阎",
  "閼":"阏",
  "閽":"阍",
  "閾":"阈",
  "閿":"阌",
  "闃":"阒",
  "闆":"板",
  "闇":"暗",
  "闈":"闱",
  "闉":"𬮱",
  "闊":"阔",
  "闋":"阕",
  "闌":"阑",
  "闍":"阇",
  "闐":"阗",
  "闑":"𫔶",
  "闒":"阘",
  "闓":"闿",
  "闔":"阖",
  "闕":"阙",
  "闖":"闯",
  "闘":"斗",
  "闚":"窥",
  "關":"关",
  "闞":"阚",
  "闡":"阐",
  "闢":"辟",
  "闥":"闼",
  "阨":"厄",
  "阬":"坑",
  "陘":"陉",
  "陝":"陕",
  "陣":"阵",
  "陥":"陷",
  "陰":"阴",
  "陳":"陈",
  "陸":"陆",
  "陽":"阳",
  "陿":"狭",
  "隄":"堤",
  "隊":"队",
  "階":"阶",
  "隑":"𬮿",
  "隕":"陨",
  "際":"际",
  "隠":"隐",
  "隣":"邻",
  "隤":"𬯎",
  "隨":"随",
  "險":"险",
  "隮":"𬯀",
  "隱":"隐",
  "隴":"陇",
  "隷":"隶",
  "隸":"隶",
  "隻":"只",
  "雋":"隽",
  "雑":"杂",
  "雖":"虽",
  "雙":"双",
  "雛":"雏",
  "雜":"杂",
  "雞":"鸡",
  "離":"离",
  "難":"难",
  "雲":"云",
  "電":"电",
  "霊":"灵",
  "霑":"沾",
  "霧":"雾",
  "霽":"霁",
  "靂":"雳",
  "靄":"霭",
  "靆":"叇",
  "靈":"灵",
  "靉":"叆",
  "靑":"青",
  "靚":"靓",
  "靜":"静",
  "靧":"𫖃",
  "靨":"靥",
  "鞀":"鼗",
  "鞏":"巩",
  "鞦":"秋",
  "鞽":"鞒",
  "韃":"鞑",
  "韆":"千",
  "韉":"鞯",
  "韋":"韦",
  "韌":"韧",
  "韍":"韨",
  "韓":"韩",
  "韙":"韪",
  "韜":"韬",
  "韞":"韫",
  "韠":"𫖒",
  "韻":"韵",
  "響":"响",
  "頁":"页",
  "頂":"顶",
  "頃":"顷",
  "項":"项",
  "順":"顺",
  "頇":"顸",
  "須":"须",
  "頊":"顼",
  "頌":"颂",
  "頍":"𫠆",
  "頎":"颀",
  "頏":"颃",
  "預":"预",
  "頑":"顽",
  "頒":"颁",
  "頓":"顿",
  "頔":"𬱖",
  "頗":"颇",
  "領":"领",
  "頜":"颌",
  "頠":"𬱟",
  "頡":"颉",
  "頤":"颐",
  "頦":"颏",
  "頫":"𫖯",
  "頭":"头",
  "頰":"颊",
  "頲":"颋",
  "頴":"颖",
  "頵":"𫖳",
  "頷":"颔",
  "頸":"颈",
  "頹":"颓",
  "頻":"频",
  "頼":"赖",
  "頽":"颓",
  "顆":"颗",
  "題":"题",
  "額":"额",
  "顎":"颚",
  "顏":"颜",
  "顒":"颙",
  "顓":"颛",
  "顔":"颜",
  "顕":"显",
  "顗":"𫖮",
  "願":"愿",
  "顙":"颡",
  "顚":"颠",
  "顛":"颠",
  "類":"类",
  "顢":"颟",
  "顣":"𫖹",
  "顥":"颢",
  "顧":"顾",
  "顫":"颤",
  "顬":"颥",
  "顯":"显",
  "顰":"颦",
  "顱":"颅",
  "顳":"颞",
  "顴":"颧",
  "風":"风",
  "颭":"飐",
  "颮":"飑",
  "颯":"飒",
  "颱":"台",
  "颳":"刮",
  "颶":"飓",
  "颸":"飔",
  "颺":"飏",
  "颼":"飕",
  "飀":"飗",
  "飄":"飘",
  "飆":"飙",
  "飛":"飞",
  "飢":"饥",
  "飦":"𫗞",
  "飩":"饨",
  "飪":"饪",
  "飫":"饫",
  "飭":"饬",
  "飮":"饮",
  "飯":"饭",
  "飲":"饮",
  "飴":"饴",
  "飼":"饲",
  "飽":"饱",
  "飾":"饰",
  "飿":"饳",
  "餃":"饺",
  "餄":"饸",
  "餅":"饼",
  "餉":"饷",
  "養":"养",
  "餌":"饵",
  "餎":"饹",
  "餏":"饻",
  "餑":"饽",
  "餒":"馁",
  "餓":"饿",
  "餔":"𫗦",
  "餕":"馂",
  "餗":"𫗧",
  "餘":"余",
  "餛":"馄",
  "餜":"馃",
  "餞":"饯",
  "餡":"馅",
  "餧":"喂",
  "館":"馆",
  "餬":"糊",
  "餱":"糇",
  "餳":"饧",
  "餶":"馉",
  "餷":"馇",
  "餼":"饩",
  "餽":"馈",
  "餾":"馏",
  "餿":"馊",
  "饁":"馌",
  "饃":"馍",
  "饅":"馒",
  "饈":"馐",
  "饉":"馑",
  "饊":"馓",
  "饋":"馈",
  "饌":"馔",
  "饑":"饥",
  "饒":"饶",
  "饗":"飨",
  "饘":"𫗴",
  "饜":"餍",
  "饞":"馋",
  "饢":"馕",
  "馬":"马",
  "馭":"驭",
  "馮":"冯",
  "馱":"驮",
  "馳":"驰",
  "馴":"驯",
  "馹":"驲",
  "馼":"𫘜",
  "駁":"驳",
  "駃":"𫘝",
  "駆":"驱",
  "駉":"𬳶",
  "駐":"驻",
  "駑":"驽",
  "駒":"驹",
  "駓":"𬳵",
  "駔":"驵",
  "駕":"驾",
  "駘":"骀",
  "駙":"驸",
  "駛":"驶",
  "駝":"驼",
  "駟":"驷",
  "駡":"骂",
  "駢":"骈",
  "駪":"𬳽",
  "駭":"骇",
  "駰":"骃",
  "駱":"骆",
  "駸":"骎",
  "駼":"𬳿",
  "駿":"骏",
  "騁":"骋",
  "騂":"骍",
  "騃":"呆",
  "騄":"𫘧",
  "騅":"骓",
  "騊":"𫘦",
  "騍":"骒",
  "騎":"骑",
  "騏":"骐",
  "騑":"𬴂",
  "験":"验",
  "騖":"骛",
  "騙":"骗",
  "騞":"𬴃",
  "騠":"𫘨",
  "騤":"骙",
  "騧":"䯄",
  "騫":"骞",
  "騭":"骘",
  "騮":"骝",
  "騰":"腾",
  "騱":"𫘬",
  "騵":"𫘪",
  "騶":"驺",
  "騷":"骚",
  "騸":"骟",
  "騾":"骡",
  "驀":"蓦",
  "驁":"骜",
  "驂":"骖",
  "驃":"骠",
  "驄":"骢",
  "驅":"驱",
  "驊":"骅",
  "驌":"骕",
  "驍":"骁",
  "驎":"𬴊",
  "驏":"骣",
  "驕":"骄",
  "驗":"验",
  "驚":"惊",
  "驛":"驿",
  "驟":"骤",
  "驢":"驴",
  "驤":"骧",
  "驥":"骥",
  "驦":"骦",
  "驩":"欢",
  "驪":"骊",
  "驫":"骉",
  "骯":"肮",
  "髏":"髅",
  "髒":"脏",
  "體":"体",
  "髕":"髌",
  "髖":"髋",
  "髪":"发",
  "髮":"发",
  "鬆":"松",
  "鬍":"胡",
  "鬚":"须",
  "鬢":"鬓",
  "鬥":"斗",
  "鬧":"闹",
  "鬨":"哄",
  "鬩":"阋",
  "鬭":"斗",
  "鬮":"阄",
  "鬱":"郁",
  "鬹":"鬶",
  "魎":"魉",
  "魘":"魇",
  "魚":"鱼",
  "魛":"鱽",
  "魢":"鱾",
  "魨":"鲀",
  "魯":"鲁",
  "魴":"鲂",
  "魷":"鱿",
  "鮀":"𬶍",
  "鮁":"鲅",
  "鮃":"鲆",
  "鮆":"𫚖",
  "鮈":"𬶋",
  "鮊":"鲌",
  "鮋":"鲉",
  "鮍":"鲏",
  "鮎":"鲇",
  "鮐":"鲐",
  "鮑":"鲍",
  "鮒":"鲋",
  "鮓":"鲊",
  "鮚":"鲒",
  "鮜":"鲘",
  "鮝":"鲞",
  "鮞":"鲕",
  "鮟":"𩽾",
  "鮠":"𬶏",
  "鮡":"𬶐",
  "鮣":"䲟",
  "鮦":"鲖",
  "鮪":"鲔",
  "鮫":"鲛",
  "鮭":"鲑",
  "鮮":"鲜",
  "鮶":"鲪",
  "鮸":"𩾃",
  "鮺":"鲝",
  "鯀":"鲧",
  "鯁":"鲠",
  "鯇":"鲩",
  "鯉":"鲤",
  "鯊":"鲨",
  "鯒":"鲬",
  "鯔":"鲻",
  "鯕":"鲯",
  "鯖":"鲭",
  "鯛":"鲷",
  "鯝":"鲴",
  "鯡":"鲱",
  "鯢":"鲵",
  "鯤":"鲲",
  "鯧":"鲳",
  "鯨":"鲸",
  "鯪":"鲮",
  "鯫":"鲰",
  "鯴":"鲺",
  "鯷":"鳀",
  "鯻":"𬶟",
  "鯽":"鲫",
  "鯿":"鳊",
  "鰁":"鳈",
  "鰂":"鲗",
  "鰃":"鳂",
  "鰆":"䲠",
  "鰈":"鲽",
  "鰉":"鳇",
  "鰊":"𬶠",
  "鰍":"鳅",
  "鰏":"鲾",
  "鰐":"鳄",
  "鰓":"鳃",
  "鰛":"鳁",
  "鰜":"鳒",
  "鰟":"鳑",
  "鰣":"鲥",
  "鰤":"𫚕",
  "鰥":"鳏",
  "鰧":"䲢",
  "鰨":"鳎",
  "鰩":"鳐",
  "鰭":"鳍",
  "鰱":"鲢",
  "鰲":"鳌",
  "鰳":"鳓",
  "鰵":"鳘",
  "鰶":"𬶭",
  "鰷":"鲦",
  "鰹":"鲣",
  "鰺":"鲹",
  "鰻":"鳗",
  "鰼":"鳛",
  "鰾":"鳔",
  "鱀":"𬶨",
  "鱂":"鳉",
  "鱅":"鳙",
  "鱇":"𩾌",
  "鱈":"鳕",
  "鱉":"鳖",
  "鱒":"鳟",
  "鱔":"鳝",
  "鱖":"鳜",
  "鱗":"鳞",
  "鱘":"鲟",
  "鱚":"𬶮",
  "鱝":"鲼",
  "鱟":"鲎",
  "鱠":"鲙",
  "鱣":"鳣",
  "鱤":"鳡",
  "鱧":"鳢",
  "鱨":"鲿",
  "鱭":"鲚",
  "鱮":"𫚈",
  "鱯":"鳠",
  "鱲":"𫚭",
  "鱸":"鲈",
  "鱺":"鲡",
  "鳥":"鸟",
  "鳧":"凫",
  "鳩":"鸠",
  "鳬":"凫",
  "鳲":"鸤",
  "鳳":"凤",
  "鳴":"鸣",
  "鳶":"鸢",
  "鳾":"䴓",
  "鴃":"𫛞",
  "鴆":"鸩",
  "鴇":"鸨",
  "鴈":"雁",
  "鴉":"鸦",
  "鴒":"鸰",
  "鴕":"鸵",
  "鴛":"鸳",
  "鴝":"鸲",
  "鴞":"鸮",
  "鴟":"鸱",
  "鴣":"鸪",
  "鴦":"鸯",
  "鴨":"鸭",
  "鴯":"鸸",
  "鴰":"鸹",
  "鴴":"鸻",
  "鴷":"䴕",
  "鴻":"鸿",
  "鴽":"𫛪",
  "鴿":"鸽",
  "鵁":"䴔",
  "鵂":"鸺",
  "鵃":"鸼",
  "鵏":"𬷕",
  "鵐":"鹀",
  "鵑":"鹃",
  "鵒":"鹆",
  "鵓":"鹁",
  "鵜":"鹈",
  "鵝":"鹅",
  "鵟":"𫛭",
  "鵠":"鹄",
  "鵡":"鹉",
  "鵪":"鹌",
  "鵬":"鹏",
  "鵮":"鹐",
  "鵯":"鹎",
  "鵲":"鹊",
  "鵾":"鹍",
  "鶄":"䴖",
  "鶇":"鸫",
  "鶉":"鹑",
  "鶊":"鹒",
  "鶏":"鸡",
  "鶓":"鹋",
  "鶖":"鹙",
  "鶘":"鹕",
  "鶚":"鹗",
  "鶠":"𬸘",
  "鶡":"鹖",
  "鶥":"鹛",
  "鶩":"鹜",
  "鶪":"䴗",
  "鶬":"鸧",
  "鶯":"莺",
  "鶱":"𬸣",
  "鶲":"鹟",
  "鶴":"鹤",
  "鶹":"鹠",
  "鶺":"鹡",
  "鶻":"鹘",
  "鶼":"鹣",
  "鷀":"鹚",
  "鷁":"鹢",
  "鷂":"鹞",
  "鷄":"鸡",
  "鷉":"䴘",
  "鷊":"鹝",
  "鷓":"鹧",
  "鷗":"鸥",
  "鷙":"鸷",
  "鷚":"鹨",
  "鷟":"𬸦",
  "鷥":"鸶",
  "鷦":"鹪",
  "鷩":"𫜁",
  "鷫":"鹔",
  "鷭":"𬸪",
  "鷯":"鹩",
  "鷲":"鹫",
  "鷳":"鹇",
  "鷸":"鹬",
  "鷹":"鹰",
  "鷺":"鹭",
  "鸇":"鹯",
  "鸊":"䴙",
  "鸌":"鹱",
  "鸏":"鹲",
  "鸑":"𬸚",
  "鸕":"鸬",
  "鸘":"鹴",
  "鸚":"鹦",
  "鸛":"鹳",
  "鸝":"鹂",
  "鸞":"鸾",
  "鹵":"卤",
  "鹹":"咸",
  "鹺":"鹾",
  "鹽":"盐",
  "麗":"丽",
  "麤":"粗",
  "麥":"麦",
  "麩":"麸",
  "麪":"面",
  "麯":"曲",
  "麴":"麹",
  "麵":"面",
  "麺":"面",
  "麽":"么",
  "黃":"黄",
  "黌":"黉",
  "黒":"黑",
  "黙":"默",
  "點":"点",
  "黨":"党",
  "黲":"黪",
  "黴":"霉",
  "黶":"黡",
  "黷":"黩",
  "黽":"黾",
  "黿":"鼋",
  "鼂":"鼌",
  "鼃":"蛙",
  "鼇":"鳌",
  "鼈":"鳖",
  "鼉":"鼍",
  "鼕":"冬",
  "齊":"齐",
  "齋":"斋",
  "齎":"赍",
  "齏":"齑",
  "齒":"齿",
  "齔":"龀",
  "齕":"龁",
  "齗":"龂",
  "齘":"𬹼",
  "齙":"龅",
  "齜":"龇",
  "齟":"龃",
  "齠":"龆",
  "齡":"龄",
  "齣":"出",
  "齦":"龈",
  "齧":"啮",
  "齪":"龊",
  "齬":"龉",
  "齮":"𬺈",
  "齯":"𫠜",
  "齲":"龋",
  "齷":"龌",
  "齼":"𬺓",
  "龍":"龙",
  "龐":"庞",
  "龑":"䶮",
  "龔":"龚",
  "龕":"龛",
  "龜":"龟",
  "廊":"廊",
  "朗":"朗",
  "虜":"虏",
  "殺":"杀",
  "類":"类",
  "隆":"隆",
  "猪":"猪",
  "益":"益",
  "神":"神",
  "祥":"祥",
  "福":"福",
  "靖":"靖",
  "精":"精",
  "羽":"羽",
  "諸":"诸",
  "都":"都",
  "飯":"饭",
  "館":"馆",
  "侮":"侮",
  "僧":"僧",
  "免":"免",
  "勉":"勉",
  "勤":"勤",
  "卑":"卑",
  "嘆":"叹",
  "器":"器",
  "墨":"墨",
  "層":"层",
  "悔":"悔",
  "慨":"慨",
  "憎":"憎",
  "懲":"惩",
  "敏":"敏",
  "既":"既",
  "梅":"梅",
  "海":"海",
  "漢":"汉",
  "煮":"煮",
  "碑":"碑",
  "社":"社",
  "祈":"祈",
  "祐":"祐",
  "祖":"祖",
  "祝":"祝",
  "禍":"祸",
  "禎":"祯",
  "穀":"谷",
  "突":"突",
  "節":"节",
  "繁":"繁",
  "署":"署",
  "者":"者",
  "臭":"臭",
  "著":"著",
  "褐":"褐",
  "視":"视",
  "謁":"谒",
  "謹":"谨",
  "賓":"宾",
  "逸":"逸",
  "難":"难",
  "頻":"频",
}
