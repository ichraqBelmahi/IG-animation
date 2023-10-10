// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <math.h>

#include <algorithm>
#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
#include "src/jmkdtree.h"




std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;

struct Triangle{
    std::vector< Vec3 >  i_positions;
    std::vector< unsigned int >  i_triangles;
}triangles;

struct Voxel{
int cube[8];
int id=-1;
};

struct Grid{
    std::vector< Vec3 > quadrillage;

    std::vector< Voxel > voxels; 
    int x=50;
    int y=50;
    int z=50;
    std::vector<std::vector<std::vector< Vec3 > > > quadrillage3d;


    Vec3 BBmin=Vec3(-1,-1,-1);    
    Vec3 BBmax=Vec3(1,1,1);
    std::vector< Vec3 > Sommet;
}grid;

// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 640;
static unsigned int SCREENHEIGHT = 480;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;


void HPSS( Vec3 inputPoint , Vec3 & outputPoint , Vec3 & outputNormal ,
std::vector<Vec3>const & positions , std::vector<Vec3>const & normals , BasicANNkdTree const & kdtree ,
int kernel_type, unsigned int nbIterations,unsigned int knn);

void initQuadrillage(Grid & grid);
void initBB(Grid & grid,std::vector<Vec3>const & positions);
void initVoxel(Grid & grid);

// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}
void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}
void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    }

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}


// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    glCullFace (GL_BACK);
    glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}



void drawTriangleMesh( std::vector< Vec3 > const & i_positions , std::vector< unsigned int > const & i_triangles ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_triangles.size() ; ++tIt) {
       
        Vec3 p0 = i_positions[3*tIt];
        Vec3 p1 = i_positions[3*tIt+1];
        Vec3 p2 = i_positions[3*tIt+2];
        Vec3 n = Vec3::cross(p1-p0 , p2-p0);
        n.normalize();
        glNormal3f( n[0] , n[1] , n[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();
}

void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}

void draw () {
    glPointSize(2); // for example...

    glColor3f(0.8,0.8,1);
    drawPointSet(positions , normals);

    glColor3f(1,0.5,0.5);
    drawPointSet(positions2 , normals2);
}





void drawQuadrillage3d(){
    glPointSize(2); 
    glColor3f(0.8,0.8,0);
    for(int i=0; i<grid.quadrillage3d.size();i++)
        for(int j=0; j<grid.quadrillage3d[i].size();j++)
           drawPointSet( grid.quadrillage3d[i][j], grid.quadrillage3d[i][j]);

}
void drawSommet(){
    glPointSize(3); 
    glColor3f(0.1,0.9,0.1);
    drawPointSet( grid.Sommet, grid.Sommet);
 
     glColor3f(1.0, 0.5, 0.5);
    drawTriangleMesh(triangles.i_positions, triangles.i_triangles);    
    
}


/*
struct Voxel {
    float x, y, z;  // Position du voxel
    float r, g, b;  // Couleur du voxel (composantes RVB)
};
void drawVoxel(const std::vector<Voxel>& voxels){
     glBegin(GL_QUADS);
    for (const Voxel& voxel : voxels) {
        glColor3f(/*.r, voxel.g, voxel.b);

        // Dessiner un cube pour chaque voxel
        glVertex3f(voxel.x - 0.5, voxel.y - 0.5, voxel.z - 0.5); // Coin inférieur gauche
        glVertex3f(voxel.x + 0.5, voxel.y - 0.5, voxel.z - 0.5); // Coin inférieur droit
        glVertex3f(voxel.x + 0.5, voxel.y + 0.5, voxel.z - 0.5); // Coin supérieur droit
        glVertex3f(voxel.x - 0.5, voxel.y + 0.5, voxel.z - 0.5); // Coin supérieur gauche
    }
    glEnd();
}*/





void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    //draw ();
    //drawQuadrillage3d();
    drawSommet();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    default:
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}


void projection(Vec3 inputPoint, Vec3 & outputPoint,const Vec3 & point, const Vec3 & normal ){
    float a = Vec3::dot(inputPoint-point, normal)/normal.length() ;
    outputPoint = inputPoint - a*normal;

}



void HPSS( Vec3 inputPoint , Vec3 & outputPoint , Vec3 & outputNormal ,
std::vector<Vec3>const & positions , std::vector<Vec3>const & normals , BasicANNkdTree const & kdtree ,
int kernel_type, unsigned int nbIterations = 1 , unsigned int knn = 10 ) {

        int k=0;
        while(k<nbIterations){

        ANNidxArray id_nearest_neighbors =new ANNidx[ knn ];
        ANNdistArray square_distances_to_neighbors = new ANNdist[ knn ];

         kdtree.knearest( inputPoint , knn , id_nearest_neighbors, square_distances_to_neighbors );

        Vec3 p = Vec3(0,0,0);
        Vec3 n =  Vec3(0,0,0);
        float sumW=0;

         Vec3 output[knn];

            for( int i=0; i<knn; i++){

        projection(inputPoint, output[i], positions[id_nearest_neighbors[i]], normals[id_nearest_neighbors[i]]);

         float h =  sqrt(square_distances_to_neighbors[knn-1]);
         double w=0;

        double r = (inputPoint - positions[id_nearest_neighbors[i]]).length();

    if (kernel_type==0){
        w=exp(-pow(r,2)/pow(h,2));
    }
        if (kernel_type==1){
        w=pow(1- ( r / h ) ,4) * ( 1 + 4 * ( r / h ) );
    }
    if (kernel_type==2){
        w=pow(h/r,2);
    }
    p += w*output[i];
    n += (w*normals[id_nearest_neighbors[i]]);
    sumW += w;

}
    outputPoint =  p / sumW;
    outputNormal = n / sumW;
  k++;
  inputPoint = outputPoint;
    delete [] id_nearest_neighbors;
    delete [] square_distances_to_neighbors;

}


}


void initBB(Grid & grid,std::vector<Vec3>const & positions  ){
    grid.BBmin=positions[0];
    grid.BBmax=positions[0];

    
    for(int i = 0; i< positions.size();i++){
        if(positions[i][0]< grid.BBmin[0])
            grid.BBmin[0]= positions[i][0];
        if(positions[i][1]< grid.BBmin[1])
            grid.BBmin[1]= positions[i][1];
        if(positions[i][2]< grid.BBmin[2])
            grid.BBmin[2]= positions[i][2];         

        if(positions[i][0]> grid.BBmax[0])
            grid.BBmax[0]= positions[i][0];
        if(positions[i][1]> grid.BBmax[1])
            grid.BBmax[1]= positions[i][1];
        if(positions[i][2]> grid.BBmax[2])
            grid.BBmax[2]= positions[i][2]; 

    }
grid.BBmin=grid.BBmin- (grid.BBmax-grid.BBmin)/10;
    grid.BBmax=grid.BBmax+ (grid.BBmax-grid.BBmin)/10;

}


void initVoxel(Grid & grid){

    for(int i=0;i<grid.x-1;i++)
        for(int j=0;j<grid.y-1;j++)
            for(int k=0;k<grid.z-1;k++){
                int count = i*grid.y*grid.z + j* grid.z + k;
                Voxel voxels;
            voxels.cube[0]=count;
            voxels.cube[1]=count+1;
            voxels.cube[2]=count+grid.z;
            voxels.cube[3]=count+grid.z+1;
            voxels.cube[4]=count+grid.z*grid.y;
            voxels.cube[5]=count+grid.z*grid.y+1;
            voxels.cube[6]=count+grid.z*grid.y+grid.z;
            voxels.cube[7]=count+grid.z*grid.y+grid.z+1;
            grid.voxels.push_back(voxels);
           
            }

}


void initQuadrillage(Grid & grid ){

    float Xmin = grid.BBmin[0];
    float Ymin = grid.BBmin[1];
    float Zmin = grid.BBmin[2];

    float Xmax = grid.BBmax[0];
    float Ymax = grid.BBmax[1];
    float Zmax = grid.BBmax[2];


   float intervalX=(Xmax-Xmin)/(float)(grid.x-1);
   float intervalY=(Ymax-Ymin)/(float)(grid.y-1);
   float intervalZ=(Zmax-Zmin)/(float)(grid.z-1);
    grid.quadrillage.resize(grid.x*grid.y*grid.z);
    
    for(int i=0;i<grid.x;i++){
        for(int j=0;j<grid.y;j++){
            for(int k=0;k<grid.z;k++){
                grid.quadrillage[i*grid.y*grid.z + j* grid.z + k] = Vec3(Xmin+intervalX*i, Ymin+intervalY*j, Zmin+intervalZ*k);
            }

        }
    }
            
}

void detectChangementSigne(Grid & grid,std::vector<Vec3>const & positions , std::vector<Vec3>const & normals , BasicANNkdTree const & kdtree){
    
    float Xmin = grid.BBmin[0];
    float Ymin = grid.BBmin[1];
    float Zmin = grid.BBmin[2];

    float Xmax = grid.BBmax[0];
    float Ymax = grid.BBmax[1];
    float Zmax = grid.BBmax[2];


   float milieuX=((Xmax-Xmin)/(float)(grid.x-1))/2;
   float milieuY=((Ymax-Ymin)/(float)(grid.y-1))/2;
   float milieuZ=((Zmax-Zmin)/(float)(grid.z-1))/2;
    for(int i = 0; i< grid.voxels.size();i++){
        bool changement =false;
        float testSigne=0;
        for(int j=0; j<8;j++){
            Vec3 outputPoint;
            Vec3 outputNormal;
            HPSS( grid.quadrillage[grid.voxels[i].cube[j]] , outputPoint , outputNormal ,positions , normals , kdtree ,  0 , 5 , 8 );
            
            float signe = Vec3::dot((grid.quadrillage[grid.voxels[i].cube[j]]-outputPoint),outputNormal);

            if((signe > 0 && testSigne <0) || (signe < 0 && testSigne > 0) )
                changement = true;
           testSigne=signe;

        }
        if(changement){
            Vec3 outputPoint=Vec3(0,0,0);
            Vec3 outputNormal=Vec3(0,0,0);
            HPSS( grid.quadrillage[grid.voxels[i].cube[0]]+Vec3(milieuX,milieuY,milieuZ) , outputPoint , outputNormal ,positions , normals , kdtree ,  0 , 5 , 8 );
            grid.voxels[i].id = grid.Sommet.size();
            grid.Sommet.push_back(outputPoint);

            
        }
    }
}

void subdiviseTriangle(Triangle & triangles,std::vector<Vec3>const & positions , std::vector<Vec3>const & normals , BasicANNkdTree const & kdtree,Vec3 pt1, Vec3 pt2, Vec3 pt3,int iteration=1){
    Vec3 milieu12= (pt1+pt2)/2.0;
    Vec3 milieu23=(pt2+pt3)/2.0;
    Vec3 milieu13=(pt3+pt1)/2.0;
    
    Vec3 outputNormal=Vec3(0,0,0);
    HPSS( milieu12 , milieu12 , outputNormal ,positions , normals , kdtree ,  0 , 5 , 8 );
    HPSS( milieu23 , milieu23 , outputNormal ,positions , normals , kdtree ,  0 , 5 , 8 );
    HPSS( milieu13 , milieu13 , outputNormal ,positions , normals , kdtree ,  0 , 5 , 8 );
    
    
    if(iteration <1){
    subdiviseTriangle(triangles,positions ,  normals ,  kdtree, pt1,  milieu12,  milieu13, iteration-1);
    subdiviseTriangle(triangles,positions ,  normals ,  kdtree, pt2,  milieu12,  milieu23, iteration-1);
        subdiviseTriangle(triangles,positions ,  normals ,  kdtree, pt3,  milieu23,  milieu13, iteration-1);
    }
    else{

        triangles.i_positions.push_back(pt1);
        triangles.i_positions.push_back(milieu12);
        triangles.i_positions.push_back(milieu13);
        triangles.i_triangles.push_back(triangles.i_triangles.size());
 
        triangles.i_positions.push_back(pt2);
        triangles.i_positions.push_back(milieu12);
        triangles.i_positions.push_back(milieu23);
        triangles.i_triangles.push_back(triangles.i_triangles.size());

        triangles.i_positions.push_back(pt3);
        triangles.i_positions.push_back(milieu23);
        triangles.i_positions.push_back(milieu13);
        triangles.i_triangles.push_back(triangles.i_triangles.size());
        
   
    
    }

}




void placeAllTriangle(Grid &grid, Triangle & triangles, std::vector<Vec3>const & positions , std::vector<Vec3>const & normals , BasicANNkdTree const & kdtree){
    
    
    for(int x=0; x<grid.voxels.size();x++){
        int pos = grid.voxels[x].cube[0];
        int list[8]= {0,1,grid.z-1,grid.z,(grid.z-1)*(grid.y-1),(grid.z-1)*(grid.y-1)+1,(grid.z-1)*(grid.y-1)+grid.z-1,(grid.z-1)*(grid.y-1)+grid.z-1+1};
        for(int a=0; a<8;a++)
            for( int b = 0; b<8; b++)
                for( int c = 0; c<8; c++)
                    if((a!=b && b != c && a != c))
                        if( (grid.voxels[pos+list[c]].id !=-1) && (grid.voxels[pos+list[a]].id !=-1) && (grid.voxels[pos+list[b]].id!=-1)){
                            subdiviseTriangle(triangles, positions ,  normals ,  kdtree,
                                              grid.Sommet[grid.voxels[pos+list[a]].id],
                                              grid.Sommet[grid.voxels[pos+list[b]].id],
                                              grid.Sommet[grid.voxels[pos+list[c]].id],50);
        
                
        }
        }}


int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("tp point processing");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutReshapeFunc (reshape);
    glutMotionFunc (motion);
    glutMouseFunc (mouse);
    key ('?', 0, 0);

    loadPN("pointsets/igea.pn" , positions , normals);
    BasicANNkdTree kdtree;
    kdtree.build(positions);
    initBB(grid,positions );
    initVoxel(grid);
    initQuadrillage(grid);
    detectChangementSigne( grid,  positions ,  normals , kdtree);
    placeAllTriangle(grid,  triangles,positions ,  normals , kdtree);
    


    glutMainLoop ();
    return EXIT_SUCCESS;
}

