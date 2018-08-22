#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;

// TODO: Carlos V Bortolotti
static line_mark *demo_line_marks;
static int demo_lines;

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    /*
       if(l.type == DETECTION){
       get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
       } else */
    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    /*
       int i,j;
       box zero = {0};
       int classes = l.classes;
       for(i = 0; i < demo_detections; ++i){
       avg[i].objectness = 0;
       avg[i].bbox = zero;
       memset(avg[i].prob, 0, classes*sizeof(float));
       for(j = 0; j < demo_frame; ++j){
       axpy_cpu(classes, 1./demo_frame, dets[j][i].prob, 1, avg[i].prob, 1);
       avg[i].objectness += dets[j][i].objectness * 1./demo_frame;
       avg[i].bbox.x += dets[j][i].bbox.x * 1./demo_frame;
       avg[i].bbox.y += dets[j][i].bbox.y * 1./demo_frame;
       avg[i].bbox.w += dets[j][i].bbox.w * 1./demo_frame;
       avg[i].bbox.h += dets[j][i].bbox.h * 1./demo_frame;
       }
    //copy_cpu(classes, dets[0][i].prob, 1, avg[i].prob, 1);
    //avg[i].objectness = dets[0][i].objectness;
    }
     */

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];

    draw_marks(display, demo_line_marks, demo_lines, demo_classes, demo_alphabet);

    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

    // TODO: Carlos V. Bortolotti
    // Verificar a colisão dos objetos e gerar arquivo com o resultado.
    check_collision(demo_line_marks, demo_lines, display, dets, nboxes, demo_thresh, demo_names, demo_classes);

    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    //demo_frame = avg_frames;
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    // TODO: Carlos V Bortolotti
    // Criar linhas de marcação
    line_mark boxes[13];

    line_mark marquesSaidaCima = {0};
    marquesSaidaCima.x = 300;
    marquesSaidaCima.y = 90;
    marquesSaidaCima.w = 20;
    marquesSaidaCima.h = 1;
    marquesSaidaCima.vertical = 0;
    marquesSaidaCima.title = "marquesSaidaCima";
    marquesSaidaCima.class_counter = calloc(demo_classes, sizeof(int));
    //marquesSaidaCima.last_cross = {0};
    boxes[0] = marquesSaidaCima;

    line_mark marquesSaidaCima2 = {0};
    marquesSaidaCima2.x = 320;
    marquesSaidaCima2.y = 90;
    marquesSaidaCima2.w = 20;
    marquesSaidaCima2.h = 1;
    marquesSaidaCima2.vertical = 0;
    marquesSaidaCima2.title = "marquesSaidaCima2";
    marquesSaidaCima2.class_counter = calloc(demo_classes, sizeof(int));
    //marquesSaidaCima.last_cross = {0};
    boxes[10] = marquesSaidaCima2;

    line_mark marquesEntradaCima = {0};
    marquesEntradaCima.x = 344;
    marquesEntradaCima.y = 90;
    marquesEntradaCima.w = 28;
    marquesEntradaCima.h = 1;
    marquesEntradaCima.vertical = 0;
    marquesEntradaCima.title = "marquesEntradaCima";
    marquesEntradaCima.class_counter = calloc(demo_classes, sizeof(int));
    //marquesEntradaCima.last_cross = {0};
    boxes[1] = marquesEntradaCima;

    line_mark marquesSaidaBaixo = {0};
    marquesSaidaBaixo.x = 308;
    marquesSaidaBaixo.y = 322;
    marquesSaidaBaixo.w = 25;
    marquesSaidaBaixo.h = 1;
    marquesSaidaBaixo.vertical = 0;    
    marquesSaidaBaixo.title = "marquesSaidaBaixo";
    marquesSaidaBaixo.class_counter = calloc(demo_classes, sizeof(int));
    //marquesSaidaBaixo.last_cross = {0};
    boxes[2] = marquesSaidaBaixo;

    line_mark marquesSaidaBaixo2 = {0};
    marquesSaidaBaixo2.x = 334;
    marquesSaidaBaixo2.y = 322;
    marquesSaidaBaixo2.w = 25;
    marquesSaidaBaixo2.h = 1;
    marquesSaidaBaixo2.vertical = 0;    
    marquesSaidaBaixo2.title = "marquesSaidaBaixo2";
    marquesSaidaBaixo2.class_counter = calloc(demo_classes, sizeof(int));
    //marquesSaidaBaixo.last_cross = {0};
    boxes[11] = marquesSaidaBaixo2;

    line_mark marquesEntradaBaixo = {0};
    marquesEntradaBaixo.x = 278;
    marquesEntradaBaixo.y = 322;
    marquesEntradaBaixo.w = 36;
    marquesEntradaBaixo.h = 1;
    marquesEntradaBaixo.vertical = 0;
    marquesEntradaBaixo.title = "marquesEntradaBaixo";
    marquesEntradaBaixo.class_counter = calloc(demo_classes, sizeof(int));
    //marquesEntradaBaixo.last_cross = {0};
    boxes[3] = marquesEntradaBaixo;

    line_mark marquesSaidaBaixoLateral = {0};
    marquesSaidaBaixoLateral.x = 398;
    marquesSaidaBaixoLateral.y = 305;
    marquesSaidaBaixoLateral.w = 45;
    marquesSaidaBaixoLateral.h = 1;
    marquesSaidaBaixoLateral.vertical = 0;
    marquesSaidaBaixoLateral.title = "marquesSaidaBaixoLateral";
    marquesSaidaBaixoLateral.class_counter = calloc(demo_classes, sizeof(int));
    //marquesSaidaBaixoLateral.last_cross = {0};
    boxes[4] = marquesSaidaBaixoLateral;

    line_mark ottoSaidaEsquerda = {0};
    ottoSaidaEsquerda.x = 170;
    ottoSaidaEsquerda.y = 213;
    ottoSaidaEsquerda.w = 1;
    ottoSaidaEsquerda.h = 26;
    ottoSaidaEsquerda.vertical = 1;
    ottoSaidaEsquerda.title = "ottoSaidaEsquerda";
    ottoSaidaEsquerda.class_counter = calloc(demo_classes, sizeof(int));
    //ottoSaidaEsquerda.last_cross = {0};
    boxes[5] = ottoSaidaEsquerda;

    line_mark ottoSaidaEsquerda2 = {0};
    ottoSaidaEsquerda2.x = 170;
    ottoSaidaEsquerda2.y = 240;
    ottoSaidaEsquerda2.w = 1;
    ottoSaidaEsquerda2.h = 26;
    ottoSaidaEsquerda2.vertical = 1;
    ottoSaidaEsquerda2.title = "ottoSaidaEsquerda2";
    ottoSaidaEsquerda2.class_counter = calloc(demo_classes, sizeof(int));
    //ottoSaidaEsquerda.last_cross = {0};
    boxes[6] = ottoSaidaEsquerda2;

    line_mark ottoSaidaDireita = {0};
    ottoSaidaDireita.x = 460;
    ottoSaidaDireita.y = 181;
    ottoSaidaDireita.w = 1;
    ottoSaidaDireita.h = 22;
    ottoSaidaDireita.vertical = 1;
    ottoSaidaDireita.title = "ottoSaidaDireita";
    ottoSaidaDireita.class_counter = calloc(demo_classes, sizeof(int));
    //ottoSaidaDireita.last_cross = {0};
    boxes[7] = ottoSaidaDireita;

    line_mark ottoSaidaDireita2 = {0};
    ottoSaidaDireita2.x = 460;
    ottoSaidaDireita2.y = 203;
    ottoSaidaDireita2.w = 1;
    ottoSaidaDireita2.h = 22;
    ottoSaidaDireita2.vertical = 1;
    ottoSaidaDireita2.title = "ottoSaidaDireita2";
    ottoSaidaDireita2.class_counter = calloc(demo_classes, sizeof(int));
    //ottoSaidaDireita.last_cross = {0};
    boxes[12] = ottoSaidaDireita2;

    line_mark ottoEntradaEsquerda = {0};
    ottoEntradaEsquerda.x = 170;
    ottoEntradaEsquerda.y = 185;
    ottoEntradaEsquerda.w = 1;
    ottoEntradaEsquerda.h = 30;
    ottoEntradaEsquerda.vertical = 1;
    ottoEntradaEsquerda.title = "ottoEntradaEsquerda";
    ottoEntradaEsquerda.class_counter = calloc(demo_classes, sizeof(int));
    //ottoEntradaEsquerda.last_cross = {0};
    boxes[8] = ottoEntradaEsquerda;

    line_mark ottoEntradaDireita = {0};
    ottoEntradaDireita.x = 415;
    ottoEntradaDireita.y = 226;
    ottoEntradaDireita.w = 1;
    ottoEntradaDireita.h = 32;
    ottoEntradaDireita.vertical = 1;
    ottoEntradaDireita.title = "ottoEntradaDireita";
    ottoEntradaDireita.class_counter = calloc(demo_classes, sizeof(int));
    //ottoEntradaDireita.last_cross = {0};
    boxes[9] = ottoEntradaDireita;

    demo_line_marks = boxes;

    int num_line_marks = sizeof(boxes)/sizeof(boxes[0]);
    demo_lines = num_line_marks;

    demo_time = what_time_is_it_now();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }

}

/*
   void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
   {
   demo_frame = avg_frames;
   predictions = calloc(demo_frame, sizeof(float*));
   image **alphabet = load_alphabet();
   demo_names = names;
   demo_alphabet = alphabet;
   demo_classes = classes;
   demo_thresh = thresh;
   demo_hier = hier;
   printf("Demo\n");
   net = load_network(cfg1, weight1, 0);
   set_batch_network(net, 1);
   pthread_t detect_thread;
   pthread_t fetch_thread;

   srand(2222222);

   if(filename){
   printf("video file: %s\n", filename);
   cap = cvCaptureFromFile(filename);
   }else{
   cap = cvCaptureFromCAM(cam_index);

   if(w){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
   }
   if(h){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
   }
   if(frames){
   cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
   }
   }

   if(!cap) error("Couldn't connect to webcam.\n");

   layer l = net->layers[net->n-1];
   demo_detections = l.n*l.w*l.h;
   int j;

   avg = (float *) calloc(l.outputs, sizeof(float));
   for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

   boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
   probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
   for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

   buff[0] = get_image_from_stream(cap);
   buff[1] = copy_image(buff[0]);
   buff[2] = copy_image(buff[0]);
   buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
   buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
   ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

   int count = 0;
   if(!prefix){
   cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
   if(fullscreen){
   cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
   } else {
   cvMoveWindow("Demo", 0, 0);
   cvResizeWindow("Demo", 1352, 1013);
   }
   }

   demo_time = what_time_is_it_now();

   while(!demo_done){
buff_index = (buff_index + 1) %3;
if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
if(!prefix){
    fps = 1./(what_time_is_it_now() - demo_time);
    demo_time = what_time_is_it_now();
    display_in_thread(0);
}else{
    char name[256];
    sprintf(name, "%s_%08d", prefix, count);
    save_image(buff[(buff_index + 1)%3], name);
}
pthread_join(fetch_thread, 0);
pthread_join(detect_thread, 0);
++count;
}
}
*/
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

