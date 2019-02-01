/*
    Example of a C++ implementation of an Iterated Local Search for TSP.

    Hyperparameters here are: T_0, ALPHA, TT_ADJUST.

    Originally used for the Kattis TSP challenge, hence the release time
    and the maximum number of vertex.
*/

#include <bits/stdc++.h>
#ifndef N_MAX 
    #define N_MAX 1005
#endif
#ifndef RELEASE_TIME
    #define RELEASE_TIME 1.97
#endif

// Settings for the look up table for the exponential
// function, see tail of file.
#define EXP_PRECISION 1000
#define EXP_MIN -10.0

using namespace std;

// Double linked list for vertex
struct vertice {
    float x;
    float y;
    int next;
    int prev;
};

// PRNG
default_random_engine re;
uniform_real_distribution<float> unif(0,1);

clock_t start;

// Graph and info about tour
vertice graph[N_MAX];
int dist[N_MAX][N_MAX], tabu_list[N_MAX][N_MAX] = {0};
int tour[N_MAX], tour_best[N_MAX];

int N, cost, cost_cur, cost_best, epoch, nplateau = 0, TT;
extern float exp_table[EXP_PRECISION];

// Temperature and decay
float T, alpha;

inline float exp_lookup(float x){
    return x <= EXP_MIN ? 0.0 : x >= 0 ? 1.0 : exp_table[(int) (x * EXP_PRECISION/EXP_MIN)];
}

/*
    Get the other vertex from `current` that is not `prev`
*/
inline int next(int prev, int cur){
    return graph[cur].prev == prev ? graph[cur].next : graph[cur].prev;
}

inline void change(vertice &v, int from, int to){
    if(v.next == from) v.next = to;
    else               v.prev = to;
}

/*
    Delta evaluation of an exchange
*/
inline int exchange_cost(int left1, int right1, int left2, int right2){
    return cost - dist[left1][right1] - dist[left2][right2] + dist[left1][right2] + dist[right1][left2];
}

inline bool tabu(int l1, int r1, int l2, int r2){
    return tabu_list[l1][r2] == tabu_list[r1][l2] && epoch - tabu_list[l1][r1] < TT;
}

void exchange(int l1, int r1, int l2, int r2){
    change(graph[l1], r1, r2);
    change(graph[r1], l1, l2);
    change(graph[l2], r2, r1);
    change(graph[r2], l2, l1);
    tabu_list[l1][r1] = epoch;
    tabu_list[r1][l1] = epoch;
    tabu_list[l2][r2] = epoch;
    tabu_list[r2][l2] = epoch;
    epoch++;
}

/*
    Replacing the best tour.
*/
void tour_collect(int tour[]){
    tour[0] = 0;
    tour[1] = graph[0].next;
    for(int i = 1; i < N-1; i++)
        tour[i+1] = next(tour[i-1], tour[i]);
}

/*
    Initialisation of the tour data-structures.
*/
void tour_init() {
    T = T_0;
    alpha = 1 - N * ALPHA;
    TT = N;
    epoch = TT + 1;
    cost = 0;
    for(int i = 0; i < N; i++) tour_best[i] = i;
    random_shuffle(tour_best, tour_best + N);
    for(int i = 0; i < N - 1; i++)  graph[tour_best[i]].next = tour_best[i+1];
    for(int i = 1; i < N; i++)      graph[tour_best[i]].prev = tour_best[i-1];
    graph[tour_best[0]].prev = tour_best[N-1];
    graph[tour_best[N-1]].next = tour_best[0];
    for(int i = 1; i < N; i++) cost += dist[tour_best[i-1]][tour_best[i]];
    cost += dist[tour_best[0]][tour_best[N-1]];
    cost_best = cost_cur = cost;
}


/*
    Intensification à la simulated annealing:
    walking the tour and choosing the perform a 2 opt if interesting.
*/
bool intensification(int start){
    tour[0] = start;
    tour[1] = graph[start].next;
    tour[2] = next(tour[0], tour[1]);
    tour[3] = next(tour[1], tour[2]);
    for(int i = 3; i < N; i++){
        int cur = exchange_cost(tour[0], tour[1], tour[i], tour[i - 1]);
        if(!tabu(tour[0], tour[1], tour[i], tour[i - 1]) && unif(re) < exp_lookup((cost - cur)/T) ){
            if(cost < cost_cur && cur > cost){
                cost_cur = cost;
                tour_collect(tour_best);
            }
            exchange(tour[0], tour[1], tour[i], tour[i - 1]);
            cost = cur;
            if(cost < cost_best){
                cost_best = cost;
                nplateau = 0;
            } else {
                nplateau++;
            }
            return true;
        }
        tour[i + 1] = next(tour[i - 1], tour[i]);
    }
    return false;
}


/*
    Reads the coordinates of the vertices
    and compute the distance matrix.
*/
void readtsp(){
    start = clock();
    scanf("%d", &N);
    for(int i = 0 ; i < N; i++)
        scanf("%f %f", &graph[i].x, &graph[i].y);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            vertice p = graph[i], q = graph[j];
            dist[i][j] = (int) (sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2)) + 0.5);
        }
    }
}

/*
    Performs sls until release time.
*/
void runsls(){
    tour_init();
    while(float(clock() - start) / CLOCKS_PER_SEC < RELEASE_TIME){
        for(int i = 0; i < N; i++)
            intensification(unif(re)*N);
        T *= alpha;
        if(nplateau > N)
            TT *= TT_ADJUST;
        else
            TT /= TT_ADJUST;
        
    }
    if(cost < cost_cur){
        cost_cur = cost;
        tour_collect(tour_best);
    }
}

int main(){
    readtsp();
    runsls();
    for(int i = 0; i < N; i++) printf("%d\n", tour_best[i]);
}

// dah big boy
float exp_table[] = {1.0,0.9900498337491681,0.9801986733067553,0.9704455335485082,0.9607894391523232,0.951229424500714,0.9417645335842487,0.9323938199059483,0.9231163463866358,0.9139311852712282,0.9048374180359595,0.8958341352965282,0.8869204367171575,0.8780954309205613,0.8693582353988059,0.8607079764250578,0.8521437889662113,0.8436648165963837,0.835270211411272,0.8269591339433623,0.8187307530779818,0.8105842459701871,0.8025187979624785,0.794533602503334,0.7866278610665535,0.7788007830714049,0.7710515858035663,0.7633794943368531,0.7557837414557255,0.7482635675785653,0.7408182206817179,0.7334469562242892,0.7261490370736909,0.7189237334319262,0.7117703227626097,0.7046880897187134,0.697676326071031,0.6907343306373547,0.6838614092123558,0.6770568744981647,0.6703200460356393,0.6636502501363194,0.6570468198150567,0.6505090947233165,0.6440364210831414,0.6376281516217733,0.631283645506926,0.6250022682827008,0.6187833918061408,0.6126263941844161,0.6065306597126334,0.6004955788122659,0.5945205479701944,0.5886049696783552,0.5827482523739896,0.5769498103804866,0.5712090638488149,0.5655254386995371,0.559898366565402,0.5543272847345071,0.5488116360940265,0.5433508690744998,0.5379444375946745,0.5325918010068972,0.5272924240430485,0.522045776761016,0.5168513344916992,0.5117085777865424,0.5066169923655895,0.5015760690660556,0.4965853037914095,0.4916441974609651,0.4867522559599717,0.48190899009020244,0.4771139155210344,0.4723665527410147,0.46766642700990924,0.46301306831122807,0.4584060113052235,0.45384479528235583,0.44932896411722156,0.4448580662229411,0.4404316545059993,0.4360492863215356,0.43171052342907973,0.4274149319487267,0.4231620823177488,0.418951549247639,0.4147829116815814,0.4106557527523455,0.4065696597405991,0.40252422403363597,0.39851904108451414,0.3945537103716011,0.39062783535852114,0.38674102345450123,0.38289288597511206,0.37908303810339883,0.37531109885139957,0.3715766910220457,0.36787944117144233,0.3642189795715233,0.3605949401730783,0.3570069605691474,0.35345468195878016,0.3499377491111553,0.3464558103300574,0.34300851741870664,0.3395955256449391,0.33621649370673334,0.33287108369807955,0.32955896107518906,0.32627979462303947,0.32303325642225295,0.31981902181630395,0.3166367693790533,0.3134861808826053,0.31036694126548503,0.30727873860113125,0.3042212640667041,0.30119421191220214,0.2981972794298874,0.2952301669240142,0.2922925776808594,0.2893842179390506,0.2865047968601901,0.2836540264997704,0.2808316217783798,0.27803730045319414,0.27527078308975234,0.2725317930340126,0.2698200563846868,0.26713530196585034,0.26447726129982396,0.261845668580326,0.2592402606458915,0.2566607769535559,0.25410695955280027,0.2515785530597565,0.24907530463166822,0.2465969639416065,0.2441432831534371,0.24171401689703645,0.23930892224375455,0.23692775868212176,0.23457028809379765,0.23223627472975883,0.22992548518672384,0.22763768838381274,0.22537265553943872,0.22313016014842982,0.2209099779593782,0.21871188695221475,0.21653566731600707,0.21438110142697794,0.21224797382674304,0.21013607120076472,0.20804518235702046,0.20597509820488344,0.20392561173421342,0.20189651799465538,0.1998876140751445,0.19789869908361465,0.19592957412690937,0.1939800422908919,0.19204990862075413,0.19013898010152055,0.1882470656387468,0.18637397603940997,0.18451952399298926,0.18268352405273466,0.1808657926171221,0.17906614791149322,0.17728440996987782,0.17552040061699686,0.17377394345044514,0.17204486382305054,0.17033298882540943,0.1686381472685955,0.1669601696670407,0.16529888822158653,0.16365413680270405,0.16202575093388075,0.16041356777517274,0.15881742610692068,0.1572371663136276,0.1556726303679973,0.1541236618151314,0.1525901057568839,0.15107180883637086,0.14956861922263506,0.14808038659546247,0.14660696213035015,0.14514819848362373,0.14370394977770293,0.1422740715865136,0.140858420921045,0.13945685621505094,0.13806923731089282,0.13669542544552385,0.1353352832366127,0.13398867466880499,0.13265546508012172,0.13133552114849312,0.1300287108784259,0.12873490358780423,0.12745396989482075,0.12618578170503877,0.12493021219858241,0.12368713581745483,0.1224564282529819,0.12123796643338168,0.12003162851145673,0.11883729385240965,0.11765484302177918,0.11648415777349697,0.11532512103806251,0.1141776169108365,0.11304153064044985,0.11191674861732888,0.11080315836233387,0.10970064851551141,0.10860910882495796,0.10752843013579495,0.1064585043792528,0.10539922456186433,0.10435048475476504,0.1033121800831002,0.10228420671553748,0.1012664618538834,0.10025884372280375,0.09926125155964566,0.09827358560436154,0.09729574708953276,0.09632763823049303,0.09536916221554961,0.09442022319630235,0.09348072627805847,0.09255057751034329,0.09162968387750484,0.09071795328941251,0.08981529457224763,0.08892161745938634,0.08803683258237255,0.0871608514619813,0.0862935864993705,0.08543495096732123,0.08458485900156469,0.08374322559219596,0.08290996657517266,0.0820849986238988,0.0812682392408917,0.08045960674953244,0.07965902028589804,0.07886639979067495,0.07808166600115317,0.07730474044329974,0.07653554542391151,0.07577400402284548,0.07502004008532698,0.07427357821433388,0.0735345437630571,0.07280286282743559,0.0720784622387661,0.07136126955638605,0.0706512130604296,0.06994822174465536,0.069252225309346,0.06856315415427791,0.06788093937176144,0.06720551273974976,0.06653680671501686,0.06587475442640295,0.06521928966812753,0.06457034689316847,0.06392786120670757,0.06329176835964073,0.06266200474215315,0.06203850737735832,0.06142121391500013,0.06081006262521797,0.06020499239237354,0.05960594270893937,0.05901285366944784,0.05842566596450083,0.057844320874838456,0.05726876026546736,0.0566989265798469,0.056134762834133725,0.05557621261148306,0.05502322005640723,0.05447572986918986,0.05393368730035602,0.053397038145197084,0.05286572873835037,0.05233970594843238,0.05181891717272583,0.05130331033191911,0.0507928338648985,0.050287436723591865,0.049787068367863944,0.04929167876046217,0.04880121836201296,0.04831563812606779,0.04783488949419837,0.04735892439114093,0.046887695219988486,0.04642115485743127,0.045959256649044204,0.04550195440462157,0.0450492023935578,0.044600955340274535,0.04415716841969286,0.04371779725275094,0.043282797901965896,0.04285212686704019,0.042425741080511385,0.04200359790344555,0.04158565512117316,0.04117187093906774,0.04076220397836621,0.04035661327203115,0.039955058260653896,0.039557498788398725,0.039163895098987066,0.03877420783172201,0.038388398017552075,0.038006427075174314,0.03762825680717622,0.03725384939621581,0.036883167401240015,0.0365161737537404,0.036152831754046426,0.0357931050676553,0.03543695772159864,0.035084354100845025,0.03473525894473856,0.03438963734347271,0.034047454734599344,0.033708676899572396,0.03337326996032608,0.03304120037588693,0.03271243493901982,0.03238694077290704,0.03206468532786077,0.03174563637806794,0.03142976201836771,0.03111703066106086,0.030807411032751076,0.030500872171217483,0.0301973834223185,0.02989691443692632,0.029599435167892,0.02930491586704076,0.029013327082197053,0.028724639654239433,0.028438824714184505,0.028155853680300106,0.027875698255247015,0.027598330423249287,0.02732372244729256,0.027051846866350416,0.026782676492638175,0.02651618440889418,0.02625234396568796,0.025991128778755347,0.02573251272635994,0.025476469946681016,0.025222974835227212,0.024972002042276155,0.024723526470339388,0.02447752327165267,0.024233967845691113,0.023992835836709175,0.023754103131304997,0.023517745856009107,0.02328374037489701,0.02305206328722557,0.02282269142509298,0.022595601851121864,0.0223707718561656,0.022148178957037315,0.02192780089426162,0.02170961562984857,0.021493601345089923,0.02127973643837717,0.021067999523041434,0.02085836942521472,0.020650825181712566,0.020445346037937653,0.02024191144580439,0.020040501061684014,0.019841094744370288,0.019643672553065292,0.01944821474538539,0.01925470177538692,0.019063114291611637,0.018873433135151486,0.018685639337732773,0.018499714119819242,0.01831563888873418,0.018133395236801075,0.017952964939502866,0.01777432995365944,0.017597472415623393,0.017422374639493515,0.01724901911534628,0.017077388507484793,0.01690746565270528,0.016739233558580632,0.016572675401761255,0.016407774526292645,0.01624451444194987,0.016082878822588433,0.015922851504511698,0.015764416484854486,0.01560755791998283,0.015452260123909515,0.015298507566725518,0.01514628487304698,0.014995576820477703,0.014846368338086832,0.014698644504901784,0.014552390548416123,0.01440759184311235,0.014264233908999256,0.014122302410163962,0.013981783153338308,0.013842662086479501,0.013704925297364945,0.013568559012200934,0.013433549594245314,0.013299883542443767,0.013167547490079751,0.013036528203437736,0.012906812580479873,0.01277838764953576,0.012651240568005305,0.012525358621074385,0.012400729220443406,0.012277339903068436,0.012155178329914935,0.012034232284723775,0.011914489672789647,0.011795938519751562,0.011678566970395442,0.011562363287468536,0.01144731585050571,0.011333413154667387,0.011220643809589084,0.011108996538242306,0.010998460175806881,0.01088902366855445,0.010780676072743084,0.010673406553522925,0.010567204383852655,0.010462058943426803,0.010357959717613696,0.010254896296404022,0.010152858373369763,0.010051835744633586,0.00995181830784842,0.009852796061187257,0.009754759102342903,0.009657697627537777,0.009561601930543505,0.009466462401710323,0.009372269527006058,0.009279013887064744,0.009186686156244664,0.009095277101695816,0.00900477758243656,0.008915178548439553,0.008826471039726723,0.00873864618547329,0.008651695203120634,0.00856560939749806,0.008480380159953269,0.00839599896749147,0.008312457381923119,0.00822974704902003,0.008147859697679989,0.008066787139099614,0.007986521265955502,0.007907054051593441,0.007828377549225773,0.007750483891136692,0.007673365287895489,0.007597014027577567,0.00752142247499327,0.007446583070924338,0.007372488331368012,0.007299130846788583,0.0072265032813764625,0.007154598372314579,0.0070834089290521185,0.007012927832585425,0.0069431480347461145,0.006874062557496249,0.006805664492230543,0.006737946999085467,0.006670903306255274,0.006604526709314811,0.006538810570549064,0.006473748318289405,0.006409333446256383,0.006345559512909116,0.006282420140801118,0.006219909015942573,0.006158019887168897,0.006096746565515638,0.006036082923599565,0.005976022895005943,0.005916560473681857,0.0058576897133356225,0.005799404726842141,0.005741699685654202,0.005684568819219595,0.005628006414404065,0.005572006814919993,0.0055165644207607716,0.005461673687640779,0.00540732912644096,0.005353525302659903,0.005300256835870402,0.005247518399181385,0.005195304718705231,0.005143610573030383,0.00509243079269919,0.005041760259690979,0.004991593906910217,0.004941926717679822,0.004892753725239476,0.004844070012248967,0.004795870710296421,0.004748150999411478,0.004700906107583276,0.004654131310283272,0.004607821929992752,0.004561973335735096,0.004516580942612666,0.004471640211348332,0.004427146647831511,0.004383095802668776,0.004339483270738895,0.00429630469075234,0.004253555744815125,0.004211232157997035,0.0041693296979041115,0.004127844174255436,0.004086771438464067,0.004046107383222199,0.00400584794209042,0.003965989089091065,0.003926526838305624,0.0038874572434761303,0.0038487763976105425,0.003810480432592037,0.0037725655187922052,0.0037350278646880674,0.003697863716482932,0.0036610693577310053,0.0036246411089657558,0.0035885753273319477,0.003552868406221362,0.0035175167749121284,0.003482516898211663,0.0034478652761031265,0.0034135584433954303,0.0033795929693767124,0.003345965457471272,0.0033126725448998926,0.003279710902343573,0.0032470772336105863,0.00321476827530687,0.003182780796509667,0.0031511115984444414,0.0031197575141649948,0.0030887154082367687,0.0030579821764233073,0.0030275547453758153,0.0029974300723258312,0.002967605144780944,0.0029380769802235503,0.002908842625812584,0.002879899158088243,0.0028512436826796323,0.0028228733340153363,0.0027947852750368437,0.002766976696914851,0.0027394448187683684,0.0027121868873866434,0.0026852001769538205,0.002658481988776367,0.0026320296510131984,0.0026058405184084983,0.00257991197202718,0.0025542414189929983,0.0025288262922292556,0.0025036640502021,0.0024787521766663585,0.002454088180413917,0.0024296695950245975,0.0024054939786195095,0.0023815589136168707,0.002357862006490233,0.002334400887529135,0.002311173210602129,0.0022881766529221693,0.002265408914814322,0.0022428677194858034,0.002220550812798294,0.0021984559630425313,0.0021765809607151255,0.002154923618297615,0.002133481770037708,0.002112253271732714,0.0020912360005151103,0.0020704278546402606,0.002049826753276235,0.002029430636295734,0.0020092374640700602,0.0019892452172651635,0.0019694518966397014,0.0019498555228451206,0.0019304541362277093,0.0019112457966326377,0.0018922285832099397,0.0018734005942224233,0.0018547599468555032,0.0018363047770289071,0.001818033239210273,0.0017999435062305911,0.0017820337691014916,0.0017643022368343355,0.0017467471362611197,0.0017293667118571557,0.0017121592255655228,0.0016951229566232505,0.0016782562013892477,0.001661557273173934,0.0016450245020705747,0.0016286562347882806,0.0016124508344866836,0.0015964066806122474,0.001580522168736217,0.0015647957103941666,0.0015492257329271562,0.001533810679324463,0.0015185490080678835,0.0015034391929775724,0.0014884797230594294,0.001473669102353996,0.0014590058497868585,0.0014444884990205433,0.0014301155983078744,0.0014158857103468033,0.0014017974121366744,0.001387849294835929,0.0013740399636212118,0.0013603680375478939,0.0013468321494119735,0.0013334309456133593,0.0013201630860205026,0.0013070272438363876,0.001294022105465848,0.0012811463703842113,0.0012683987510072388,0.0012557779725623694,0.0012432827729612404,0.001230911902673481,0.0012186641246017523,0.0012065382139580404,0.0011945329581411752,0.0011826471566155727,0.0011708796207911744,0.0011592291739045914,0.0011476946509014264,0.0011362748983197658,0.0011249687741748374,0.0011137751478448032,0.0011026928999577027,0.0010917209222795108,0.0010808581176033184,0.0010701033996396044,0.0010594556929076101,0.0010489139326277882,0.0010384770646153282,0.0010281440451747298,0.0010179138409954387,0.0010077854290485105,0.000997757796484312,0.0009878299405312295,0.0009780008683953946,0.0009682695971614017,0.0009586351536940199,0.0009490965745408727,0.0009396529058360961,0.000930303203204949,0.0009210465316693785,0.0009118819655545162,0.0009028085883961136,0.0008938254928488935,0.0008849317805958146,0.0008761265622582417,0.0008674089573070025,0.0008587780939743372,0.0008502331091667194,0.000841773148378549,0.0008333973656066964,0.0008251049232659046,0.0008168949921050283,0.0008087667511241114,0.0008007193874922814,0.0007927520964664691,0.0007848640813109316,0.0007770545532175816,0.0007693227312271008,0.0007616678421508473,0.0007540891204935333,0.0007465858083766792,0.0007391571554628196,0.0007318024188804728,0.0007245208631498506,0.0007173117601093135,0.000710174388842549,0.0007031080356064829,0.0006961119937599026,0.0006891855636927931,0.0006823280527563766,0.0006755387751938444,0.0006688170520717824,0.0006621622112122764,0.0006555735871256958,0.0006490505209441411,0.0006425923603555579,0.0006361984595385052,0.000629868179097574,0.0006236008859994445,0.000617395953509584,0.0006112527611295723,0.0006051706945350532,0.0005991491455142981,0.000593187511907387,0.0005872851975459907,0.0005814416121937556,0.0005756561714872761,0.0005699282968776604,0.0005642574155726738,0.0005586429604794611,0.0005530843701478336,0.0005475810887141261,0.000542132565845609,0.0005367382566854548,0.0005313976217982529,0.0005261101271160638,0.0005208752438850126,0.0005156924486124138,0.0005105612230144218,0.0005054810539642003,0.0005004514334406108,0.0004954718584774093,0.0004905418311129505,0.0004856608583403892,0.00048082845205838073,0.00047604412902226933,0.00047130741079576537,0.0004666178237030984,0.0004619748987816513,0.0004573781717350623,0.00045282718288679695,0.00044832147713417755,0.000443860603902874,0.00043944411710184546,0.0004350715750787321,0.00043074254057568753,0.00042645658068565383,0.00042221326680907034,0.0004180121746110129,0.00041385288397876164,0.0004097349789797868,0.00040565804782015684,0.0004016216828033581,0.00039762548028952584,0.0003936690406550783,0.0003897519682527548,0.00038587387137205064,0.0003820343622000467,0.00037823305678262576,0.00037446957498607833,0.0003707435404590882,0.0003670545805950984,0.0003634023264950478,0.00035978641293048306,0.00035620647830703403,0.00035266216462825575,0.0003491531174598264,0.00034567898589410494,0.0003422394225150394,0.0003388340833634261,0.00033546262790251185,0.00033212471898394095,0.00032882002281403995,0.00032554820892043796,0.0003223089501190191,0.00031910192248120326,0.00031592680530155527,0.00031278328106571065,0.0003096710354186262,0.0003065897571331437,0.0003035391380788668,0.0003005188731913479,0.00029752866044158134,0.0002945682008057998,0.0002916371982355737,0.000288735359628203,0.0002858623947974087,0.0002830180164443136,0.000280201940128712,0.0002774138842406257,0.00027465356997214254,0.00027192072128953476,0.00026921506490565784,0.00026653633025261806,0.00026388424945471793,0.00026125855730166754,0.0002586589912220635,0.00025608529125713156,0.0002535372000347304,0.00025101446274361446,0.00024851682710795185,0.00024604404336209853,0.0002435958642256188,0.00024117204487855885,0.00023877234293696414,0.00023639651842864072,0.00023404433376915794,0.00023171555373808968,0.00022940994545549173,0.00022712727835861536,0.0002248673241788482,0.00022262985691888897,0.00022041465283014714,0.00021822149039036782,0.0002160501502814794,0.00021390041536766149,0.00021177207067363092,0.0002096649033631454,0.00020757870271771752,0.00020551326011554428,0.00020346836901064417,0.0002014438249122027,0.00019943942536412282,0.00019745496992477945,0.0001954902601469749,0.00019354509955809383,0.00019161929364045703,0.00018971264981186754,0.00018782497740635362,0.0001859560876551017,0.0001841057936675792,0.00018227391041284545,0.00018046025470104844,0.00017866464516510524,0.0001768869022425666,0.00017512684815765842,0.00017338430690350557,0.00017165910422453046,0.0001699510675990275,0.00016826002622191084,0.00016658581098763354,0.00016492825447327666,0.00016328719092180806,0.00016166245622550477,0.0001600538879095432,0.00015846132511575126,0.00015688460858652242,0.00015532358064888985,0.00015377808519875894,0.00015224796768529672,0.0001507330750954765,0.00014923325593877739,0.00014774836023203364,0.00014627823948443712,0.000144822746682688,0.00014338173627629318,0.00014195506416301115,0.0001405425876744417,0.00013914416556175865,0.0001377596579815859,0.0001363889264820114,0.00013503183398874292,0.00013368824479140023,0.0001323580245299439,0.00013104104018123928,0.00012973716004575404,0.0001284462537343878,0.0001271681921554341,0.00012590284750166984,0.0001246500932375751,0.00012340980408667956,0.0001221818560190345,0.00012096612623880994,0.00011976249317201468,0.00011857083645433903,0.00011739103691911796,0.00011622297658541522,0.00011506653864622381,0.00011392160745678613,0.00011278806852302912,0.00011166580849011478,0.0001105547151311046,0.00010945467733573676,0.00010836558509931485,0.000107287329511708,0.00010621980274645875,0.00010516289805000093,0.00010411650973098415,0.00010308053314970452,0.00010205486470764058,0.00010103940183709342,0.00010003404299092957,9.903868763242697e-05,9.805323622522013e-05,9.707759022334712e-05,9.61116520613947e-05,9.515532514474172e-05,9.420851383989959e-05,9.32711234649488e-05,9.234306028007071e-05,9.142423147817327e-05,9.051454517561092e-05,8.961391040299518e-05,8.872223709609823e-05,8.783943608684635e-05,8.696541909440292e-05,8.610009871634035e-05,8.524338841989974e-05,8.439520253333736e-05,8.355545623735804e-05,8.272406555663223e-05,8.190094735139904e-05,8.1086019309152e-05,8.027919993640778e-05,7.948040855055677e-05,7.86895652717947e-05,7.790659101513453e-05,7.713140748249839e-05,7.636393715488688e-05,7.56041032846277e-05,7.48518298877006e-05,7.4107041736139e-05,7.336966435050708e-05,7.263962399245182e-05,7.191684765732902e-05,7.120126306690273e-05,7.049279866211783e-05,6.979138359594336e-05,6.909694772628816e-05,6.840942160898655e-05,6.77287364908539e-05,6.705482430281112e-05,6.638761765307784e-05,6.572704982043295e-05,6.507305474754294e-05,6.442556703435542e-05,6.378452193155948e-05,6.314985533411063e-05,6.252150377482026e-05,6.189940441800879e-05,6.128349505322213e-05,6.0673714089010444e-05,6.007000054676936e-05,5.947229405464145e-05,5.888053484147942e-05,5.829466373086881e-05,5.771462213521033e-05,5.7140352049861057e-05,5.657179604733388e-05,5.600889727155477e-05,5.5451599432176945e-05,5.489984679895225e-05,5.4353584196157486e-05,5.381275699707714e-05,5.327731111854062e-05,5.274719301551385e-05,5.2222349675744776e-05,5.1702728614462046e-05,5.118827786912642e-05,5.067894599423484e-05,5.017468205617528e-05,4.967543562813372e-05,4.918115678505129e-05,4.869179609863181e-05,4.820730463239883e-05,4.772763393680197e-05,4.725273604437187e-05,4.67825634649237e-05,4.631706918080762e-05,4.585620664220731e-05};