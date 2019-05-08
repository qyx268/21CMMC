#include "../Parameter_files/INIT_PARAMS.H"
//#include "../Parameter_files/ANAL_PARAMS.H"
//#include "../Parameter_files/Variables.h"
#include "bubble_helper_progs.c"
#include "heating_helper_progs.c"
#include "gsl/gsl_sf_erf.h"
#include "filter.c"
// below two lines are TEST
#include <stdlib.h>
#include <time.h>
#ifdef USE_KERAS
#include "keras_model.h"
#endif
/* 
 
 This is the main file for 21CMMC. This combines Ts.c, find_HII_bubbles.c, delta_T.c and redshift_interpolate_boxes.c (additionally, it includes init.c and perturb_field.c for varying the cosmology) 
 from 21cmFAST
 
 Author: Brad Greig (July 7th 2017). This is effectively an entire re-write of 21cmFAST.
 
 It is called from command line, with a fixed number of arguments (order is important). There is basically no error checking, as it would be too complicated to use that within the MCMC.
 
 An example command line call: ./drive_21cmMC_streamlined 1.000000 1.000000 0 1 1 6.0 1 
 
 First two indices are required for opening the Walker_ID1_ID2.txt and WalkerCosmology_ID1_ID2.txt which contain all the cosmology and astrophysical parameters (example included)
 
 Third argument: (0 or N), contains the number of co-eval redshifts to be sampled (will be ignored if the light-cone option is set)

 Fourth argument: (0 or 1), calculate the light-cone (1) or co-eval (0) redshifts

 Fifth argument: (0 or 1), whether or not to include the new parametrization for the ionising efficiency.
 
 Sixth argument: Redshift to which Ts.c is evolved down to

 Seventh argument: (0 or 1), calculates luminosity functions (1) or not (0). (New in v1.4)

 
*/

/* Throughout this and other 21cmMC drivers, the use of Deltac is not for checking against
 the z=0 collapsed fraction, but rather, it is used as a threshold for the validity of the
 collapse fraction expression. This expression is only valid up to Deltac
 */

// For storing the 21cm PS light cone filenames to be able to write them to file to be read by the MCMC sampler
char lightcone_box_names[1000][500];

float REDSHIFT;
#ifdef USE_KERAS
double REDSHIFT_norm;
#endif

void init_21cmMC_Ts_arrays();
void init_21cmMC_Ts_save_fcoll(); // New in v1.4
void init_21cmMC_HII_arrays();
void init_21cmMC_TsSaveBoxes_arrays();
void init_LF_arrays(); // New in v1.4

void ComputeBoxesForFile();
void ComputeTsBoxes();
float ComputeIonisationBoxes(int sample_index, float REDSHIFT_SAMPLE, float PREV_REDSHIFT);

void adj_complex_conj();
void ComputeInitialConditions();
void ComputePerturbField(float REDSHIFT_SAMPLE);
void GeneratePS(int CO_EVAL, double AverageTb);

void ComputeLF(); // New in v1.4

void ReadFcollTable();

void destroy_21cmMC_Ts_arrays();
void destroy_21cmMC_Ts_save_fcoll(); // New in v1.4
void destroy_21cmMC_HII_arrays(int skip_deallocate);
void destroy_21cmMC_TsSaveBoxes_arrays();
void destroy_LF_arrays(); // New in v1.4

int USE_FFTW_WISDOM = 1;

#ifdef USE_KERAS
double FcollzX_val_emulator(double f_star10_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm);
#endif

// This, and the one below are functions for determining the correct cell positions for direction of the light-cone. Tested this for the z-direction, but should be valid for all.
// Note that there is no option for FLIP_BOXES as we want to mimic the observed light-cone
unsigned long long coeval_box_pos(int LOS_dir,int xi,int yi,int zi){
    unsigned long long position;
    
    switch(LOS_dir) {
        case 0:
            position = HII_R_INDEX(zi, xi, yi);
            break;
        case 1:
            position = HII_R_INDEX(xi, zi, yi);
            break;
        case 2:
            position = HII_R_INDEX(xi, yi, zi);
            break;
    }
    return position;
}

unsigned long long coeval_box_pos_FFT(int LOS_dir,int xi,int yi,int zi){
    unsigned long long position;
    
    switch(LOS_dir) {
        case 0:
            position = HII_R_FFT_INDEX(zi, xi, yi);
            break;
        case 1:
            position = HII_R_FFT_INDEX(xi, zi, yi);
            break;
        case 2:
            position = HII_R_FFT_INDEX(xi, yi, zi);
            break;
    }
    return position;
}

int main(int argc, char ** argv){
    
//    printf("begin, time=%06.2f min\n", (double)clock()/CLOCKS_PER_SEC/60.0);
    
    // The standard build of 21cmFAST requires openmp for the FFTs. 21CMMC does not, however, for some computing architectures, I found it important to include this
    omp_set_num_threads(1);
    flag_generate_tables = 0;
    
    char filename[500];
    char dummy_string[500];
    FILE *F;
    
    LC_BOX_PADDING = (int)ceil(LC_BOX_PADDING_IN_MPC/((float)BOX_LEN*(float)HII_DIM));
    
    int i,j,k,temp_int,temp_int2, counter;
    float z_prime,prev_z_prime;
    float nf_ave;
    
    unsigned long long ct;

    
    // Setting it to its maximum value
    INHOMO_RECO_R_BUBBLE_MAX = 50.0;
    
    // All parameters, redshifts and flag options are written to file by the python MCMC driver

    // Take as input:
    // 1) Random ID for MCMC walker
    // 2) Number of parameters set by the user
    // 3) Number of redshifts required (number of co-eval cubes to be used. Will be overwritten if the lightcone option is set)
    
    // Assign random walker ID (takes two values as a fail-safe. First is a random number generated by the MCMC algorithm, the second is the first parameter of the specific walker)
    INDIVIDUAL_ID = atof(argv[1]);
    INDIVIDUAL_ID_2 = atof(argv[2]);
    
    // Number of user defined redshifts for which find_HII_bubbles will be called
    N_USER_REDSHIFT = atof(argv[3]);

    // Flag set to 1 if light cone boxes are to be used (feature has yet to be added)
    USE_LIGHTCONE = atof(argv[4]);
    
    // ****** NOTE: Need to add in a flag here, which toggles including alpha as a useable parameter ****** //
    // ****** In doing it here, it enables the majority of the remaining code to be relatively straight-forward (i.e. doesn't change existing text-file structure etc.) ****** //
    // ****** This hasn't been rigorously checked yet. Need to look into this at some point... ******* //
    //INCLUDE_ZETA_PL = atof(argv[5]);
    USE_MASS_DEPENDENT_ZETA = atoi(argv[5]); // New in v1.4
    
    // Redshift for which Ts.c is evolved down to, i.e. z'
    REDSHIFT = atof(argv[6]);
#ifdef USE_KERAS
	REDSHIFT_norm = ( (double) REDSHIFT - HEIGHT_REDSHIFT ) / WIDTH_REDSHIFT + CENTER_REDSHIFT;
#endif

    // New in v1.4
    // Flag set to 1 if Luminosity functions are to be used together with outputs from 21cm signals.
    // Flag set to 2 if one wants to compute Luminosity functions with tau_e, i.e. without PS
    USE_LF = atof(argv[7]);

    // Determines the lenght of the walker file, given the values set by TOTAL_AVAILABLE_PARAMS in Variables.h and the number of redshifts
    if(USE_LIGHTCONE) {
        WALKER_FILE_LENGTH = TOTAL_AVAILABLE_PARAMS + 1;
    }
    else {
        WALKER_FILE_LENGTH = N_USER_REDSHIFT + TOTAL_AVAILABLE_PARAMS + 1;
    }
    
    // Create arrays to read in all the parameter data from the two separate walker files
    double *PARAM_COSMOLOGY_VALS = (double *) calloc(TOTAL_COSMOLOGY_FILEPARAMS,sizeof(double));
    double *PARAM_VALS = (double*) calloc(TOTAL_AVAILABLE_PARAMS,sizeof(double));
    
    /////////////////   Read in the cosmological parameter data     /////////////////
    sprintf(filename,"WalkerCosmology_%1.6lf_%1.6lf.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
    F = fopen(filename,"rt");
    
    for(i=0;i<TOTAL_COSMOLOGY_FILEPARAMS;i++) {
        fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_COSMOLOGY_VALS[i]);
    }
    fclose(F);
    
    // Assign these values. Hard-coded, so order is important
    RANDOM_SEED = (unsigned long long)PARAM_COSMOLOGY_VALS[0];
    SIGMA8 = (float)PARAM_COSMOLOGY_VALS[1];
#ifdef USE_KERAS
	SIGMA8_norm = ( (double) PARAM_COSMOLOGY_VALS[1] - HEIGHT_SIGMA8 ) / WIDTH_SIGMA8 + CENTER_SIGMA8;
#endif
    hlittle = (float)PARAM_COSMOLOGY_VALS[2];
    OMm = (float)PARAM_COSMOLOGY_VALS[3];
    OMl = (float)PARAM_COSMOLOGY_VALS[4];
    OMb = (float)PARAM_COSMOLOGY_VALS[5];
    POWER_INDEX = (float)PARAM_COSMOLOGY_VALS[6]; //power law on the spectral index, ns
    //printf("sig8 = %.4f, hlittle = %.4f, OMm = %.4f, OMl = %.4f, OMb = %.4f, POWER_INDEX = %.4f\n",SIGMA8, hlittle, OMm, OMl, OMb, POWER_INDEX);
    
    
    
    /////////////////   Read in the astrophysical parameter data     /////////////////
    
    // Determine length of parameter file to read in
    // All available parameters to be varied in the MCMC are always listed, but are toggled on/off using 1/0
    // The MCMC sets the toggle, this C file reads the toggle and uses/sets the parameter values appropriately
    
    sprintf(filename,"Walker_%1.6lf_%1.6lf.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
    F = fopen(filename,"rt");
    
    if(!INHOMO_RECO||!USE_LIGHTCONE) {
        redshifts = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
    }
    
    temp_int = 0;
    temp_int2 = 0;
    for(i=0;i<WALKER_FILE_LENGTH;i++) {
        if(i==0) {
            fscanf(F,"%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",&dummy_string,&GenerateNewICs,&SUBCELL_RSD,&USE_FCOLL_IONISATION_TABLE,&SHORTEN_FCOLL,&USE_TS_FLUCT,&INHOMO_RECO,&STORE_DATA);
        }
        else if (i > 0 && i <= TOTAL_AVAILABLE_PARAMS) {
            fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_VALS[temp_int]);
            temp_int += 1;
        }
        else {
            if(!INHOMO_RECO) {
                fscanf(F,"%s\t%lf\n",&dummy_string,&redshifts[temp_int2]);
                temp_int2 += 1;
            }
            else {
                // Since the INHOMO_RECO flag has been set, we need to take all the redshifts that are used for the Ts.c part of the calculation (in order to have the right
                // time-steps for tracking the recombinations.
                continue;
            }
        }
    }
    fclose(F);
    
    
    // GenerateNewICs: Whether to create a new density field at each sampling (i.e. new initial conditions). Must use if the cosmology is being varied
    // SUBCELL_RSD: Whether to include redshift space distortions along the line-of-sight (z-direction only).
    // USE_FCOLL_IONISATION_TABLE: Whether to use an interpolation for the collapsed fraction for the find_HII_bubbles part of the computation
    // SHORTEN_FCOLL: Whether to use an interpolation for the collapsed fraction for the Ts.c computation
    // USE_TS_FLUCT: Whether to perform the full evolution of the IGM spin temperature, or just assume the saturated spin temperature limit
    // INHOMO_RECO: Whether to include inhomogeneous recombinations into the calculation of the ionisation fraction
    // STORE_DATA: Whether to output the global data for the IGM neutral fraction and average temperature brightness (used for the global signal)
    

    // Initialise the power spectrum data, and relevant functions etc., for the entire file here (i.e. it is only done once here)
    init_ps();

    
    // If the USE_LIGHTCONE option is set, need to determing the size of the entire line-of-sight dimension for storing the slice indexes and corresponding reshifts per slice
    dR = (BOX_LEN / (double) HII_DIM) * CMperMPC; // size of cell (in comoving cm)
    
    if(USE_LIGHTCONE||INHOMO_RECO) {
        // Determine the number of redshifts within the Ts.c calculation to set N_USER_REDSHIFT for the light-cone version of the computation.
        
        counter = 0;
        
        z_prime = REDSHIFT*1.0001; //higher for rounding
        while (z_prime < Z_HEAT_MAX)
            z_prime = ((1.+z_prime)*ZPRIME_STEP_FACTOR - 1.);
        prev_z_prime = Z_HEAT_MAX;
        z_prime = ((1.+z_prime)/ ZPRIME_STEP_FACTOR - 1.);
        
        while (z_prime > REDSHIFT){
            counter += 1;
            
            prev_z_prime = z_prime;
            z_prime = ((1.+prev_z_prime) / ZPRIME_STEP_FACTOR - 1.);
            
        }
        
        N_USER_REDSHIFT = counter;
        // Number of redshifts for boxes used to construct the light-cone. Light-cone ends at final redshift, final box of light-cone is the penultimate box linear interpolated to the final redshift
        N_USER_REDSHIFT_LC = counter - 1;
        
        if(USE_LIGHTCONE) {
            redshifts_LC = (double*) calloc(N_USER_REDSHIFT_LC,sizeof(double));
            start_index_LC = (int*) calloc(N_USER_REDSHIFT_LC,sizeof(int));
            end_index_LC = (int*) calloc(N_USER_REDSHIFT_LC,sizeof(int));
        }
    }
    
    // Hard coded to 100,000. Should never excede this, unless very high resolution boxes are being used! (200^3, from z_min = 6 to z_max (z = 35) corresponds to 2232 indices).
    full_index_LC = (int*) calloc(100000,sizeof(int));
    slice_redshifts = (double*) calloc(100000,sizeof(double));
    
    if(INHOMO_RECO||USE_LIGHTCONE) {
        redshifts = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
    }
    
    
    // Some very rudimentary conditionals (the python script should catch them. But, add them here in the C code for testing purposes
    if(GenerateNewICs==1 && (USE_FCOLL_IONISATION_TABLE==1 || SHORTEN_FCOLL==1)) {
        printf("\n");
        printf("Cannot use interpolation tables when generating new initial conditions on the fly\n");
        printf("(Interpolation tables are only valid for a single cosmology/initial condition)\n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }
    
    
//    In the new version support this option.
//    if(USE_TS_FLUCT==1 && INCLUDE_ZETA_PL==1) {
//        printf("\n");
//        printf("Cannot use a non-constant ionising efficiency (zeta) in conjuction with the IGM spin temperature part of the code.\n");
//        printf("(This will be changed in future)\n");        
//        printf("\n");
//        printf("Exiting...");
//        printf("\n");
//        // Probably should do free the memory properly here...
//        return 0;
//    }
    

    if(USE_FCOLL_IONISATION_TABLE==1 && INHOMO_RECO==1) {
        printf("\n");
        printf("Cannot use the f_coll interpolation table for find_hii_bubbles with inhomogeneous recombinations\n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }
    
    if(INHOMO_RECO==1 && USE_TS_FLUCT==0) {
        printf("\n");
        printf("Inhomogeneous recombinations have been set, but the spin temperature is not being computed.\n");
        printf("Inhomogeneous recombinations can only be used in combination with the spin temperature calculation (different from 21cmFAST).\n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }

    if(USE_FCOLL_IONISATION_TABLE==1 && USE_MASS_DEPENDENT_ZETA==1) {
        printf("\n");
        printf("Current version does not support an interpolation for the collapsed fraction for the halo mass-dependent ionizing efficiency parametrization. \n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }   
    
#ifdef MINI_HALO
    if(USE_FCOLL_IONISATION_TABLE==1 || GenerateNewICs==1 || SHORTEN_FCOLL==1 || USE_MASS_DEPENDENT_ZETA==0 || INHOMO_RECO==0 || USE_TS_FLUCT==0) {
        printf("\n");
        printf("ifdef MINI_HALO, you need to set USE_FCOLL_IONISATION_TABLE=0, GenerateNewICs=0, SHORTEN_FCOLL=0, USE_MASS_DEPENDENT_ZETA=1, INHOMO_RECO=1, and USE_TS_FLUCT=1. \n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }
    LOG10MTURN_INT = (double) ((LOG10MTURN_MAX+9e-8 - LOG10MTURN_MIN)) / ((double) (LOG10MTURN_NUM - 1.));
#endif
    
    ///////////////// Hard coded assignment of parameters, but can't do much about it (problem of merging C and Python code) //////////////////////////////////
    // Constant ionizing efficiency parameter
    HII_EFF_FACTOR = PARAM_VALS[6];
    // New in v1.4
    // Halo mass dependent ionizing efficiency parametrization.
    F_STAR10 = pow(10.,PARAM_VALS[0]);
    ALPHA_STAR = PARAM_VALS[1];
    F_ESC10 = pow(10.,PARAM_VALS[2]);
    ALPHA_ESC = PARAM_VALS[3];
#ifdef USE_KERAS
	F_STAR10_norm = ( (double) PARAM_VALS[0] - HEIGHT_F_STAR10 ) / WIDTH_F_STAR10 + CENTER_F_STAR10;
	ALPHA_STAR_norm = ( (double) PARAM_VALS[1] - HEIGHT_ALPHA_STAR ) / WIDTH_ALPHA_STAR + CENTER_ALPHA_STAR;
	F_ESC10_norm = ( (double) PARAM_VALS[2] - HEIGHT_F_ESC10 ) / WIDTH_F_ESC10 + CENTER_F_ESC10;
	ALPHA_ESC_norm = ( (double) PARAM_VALS[3] - HEIGHT_ALPHA_ESC ) / WIDTH_ALPHA_ESC + CENTER_ALPHA_ESC;
#endif
    M_TURN = pow(10.,PARAM_VALS[4]);
    t_STAR = PARAM_VALS[5];
   
    // New in v1.4
    if(USE_MASS_DEPENDENT_ZETA) ION_EFF_FACTOR = N_GAMMA_UV * F_STAR10 * F_ESC10;
    else ION_EFF_FACTOR = HII_EFF_FACTOR;
    
#ifdef MINI_HALO
    initialiseSplinedSigmaM_quicker(1e5/50.,1e20);
#else
    initialiseSplinedSigmaM_quicker(M_TURN/50.,1e20);
#endif
    
    // If inhomogeneous recombinations are set, need to switch to an upper limit on the maximum bubble horizon (this is set above).
    // The default choice is chosen to be 50 Mpc, as is default in 21cmFAST.
    if(INHOMO_RECO) {
        R_BUBBLE_MAX = INHOMO_RECO_R_BUBBLE_MAX;
    }
    else {
        R_BUBBLE_MAX = PARAM_VALS[7];
    }

    ION_Tvir_MIN = pow(10.,PARAM_VALS[8]);
    L_X = pow(10.,PARAM_VALS[9]);
    NU_X_THRESH = PARAM_VALS[10];
    NU_X_BAND_MAX = PARAM_VALS[11];
    NU_X_MAX = PARAM_VALS[11];
    X_RAY_SPEC_INDEX = PARAM_VALS[13];
    X_RAY_Tvir_MIN = pow(10.,PARAM_VALS[14]);
    X_RAY_Tvir_LOWERBOUND = PARAM_VALS[15];
    X_RAY_Tvir_UPPERBOUND = PARAM_VALS[16];
    //F_STAR = PARAM_VALS[16];
    N_RSD_STEPS = (int)PARAM_VALS[17];
    LOS_direction = (int)PARAM_VALS[18]; 
    
    // Converts the variables from eV into the quantities that Ts.c is familiar with
    NU_X_THRESH *= NU_over_EV;
    NU_X_BAND_MAX *= NU_over_EV;
    NU_X_MAX *= NU_over_EV;

    // New in v1.5
#ifdef MINI_HALO
    F_STAR10_MINI = pow(10.,PARAM_VALS[20]) * pow(1e3, ALPHA_STAR);
#ifdef USE_KERAS
	F_STAR7_MINI_norm = ( (double) PARAM_VALS[20] - HEIGHT_F_STAR7_MINI ) / WIDTH_F_STAR7_MINI + CENTER_F_STAR7_MINI;
#endif
    F_ESC_MINI = pow(10.,PARAM_VALS[21]);
    L_X_MINI = L_X;//pow(10.,PARAM_VALS[22]);
    X_RAY_SPEC_INDEX_MINI = X_RAY_SPEC_INDEX; //PARAM_VALS[23];
    F_H2_SHIELD = pow(10., PARAM_VALS[24]);
    if (F_H2_SHIELD > 1) F_H2_SHIELD = 1.0;
    if (F_H2_SHIELD < 0) F_H2_SHIELD = 0.0;
    ION_EFF_FACTOR_MINI = N_GAMMA_UV_MINI * F_STAR10_MINI * F_ESC_MINI;
#endif

#ifndef MINI_HALO
    ////////////////// Compute luminosity functions ////////////////////////////////////////////////
    // New in v1.4
    if (USE_LF) {
    // Compute luminosity functions using the parametrization of SFR.
        init_LF_arrays();

        ComputeLF();

        destroy_LF_arrays();
    }
#endif
    
    /////////////////   Populating requisite arrays for the construction of the light-cone box (including the indexing and individual slice redshifts etc.     /////////////////
    
    // Direction for LOS for light-cone (0 = x, 1 = y, 2 = z). Store the original starting LOS_direction
    Original_LOS_direction = LOS_direction;
    // This is set to 2 (z-direction) for the case when USE_FCOLL_IONISATION_TABLE but the full box is required anyway (i.e. cannot reduce the computation for the light-cone code).
    Default_LOS_direction = 2;
    
    Stored_LOS_direction_state_1 = LOS_direction;

    if(USE_LIGHTCONE||INHOMO_RECO) {
        
        counter = 0;
        
        z_prime = REDSHIFT*1.0001; //higher for rounding
        while (z_prime < Z_HEAT_MAX)
            z_prime = ((1.+z_prime)*ZPRIME_STEP_FACTOR - 1.);
        prev_z_prime = Z_HEAT_MAX;
        z_prime = ((1.+z_prime)/ ZPRIME_STEP_FACTOR - 1.);

        while (z_prime > REDSHIFT){
            redshifts[counter] = z_prime;
            
            counter += 1;
            
            prev_z_prime = z_prime;
            z_prime = ((1.+prev_z_prime) / ZPRIME_STEP_FACTOR - 1.);
        }
        
        if(USE_LIGHTCONE) {

            // For determining parameters for the interpolation of the boxes for the construction of the light-cone
            // redshift_interpolate_boxes.c required redshifts in increasing order, the array redshifts is in decreasing order. Therefore just invert to leave as much code the same as possible
            z1_LC = redshifts[N_USER_REDSHIFT-1];
        
            z_LC = start_z = z1_LC;
            slice_ct = 0;
            total_slice_ct = 0;
            num_boxes_interp = 0;
            i = 0;

            while(z1_LC < redshifts[0]) {
                z2_LC = redshifts[N_USER_REDSHIFT-2-i];
                // now do the interpolation
                while (z_LC < z2_LC){ // until we move to the next set of boxes
                    slice_redshifts[total_slice_ct] = z_LC;
                    full_index_LC[total_slice_ct] = total_slice_ct;
                    // check if we filled-up our array and write-out
                    if (slice_ct == HII_DIM){
                        end_z = z_LC;
                        num_boxes_interp += 1;
                
                        // update quantities
                        slice_ct=0;
                        start_z = end_z;
                    
                    } // we are now continuing with a new interpolation box
                
                    slice_ct++;
                    total_slice_ct++;
                    z_LC -= dR / drdz(z_LC);
                } // done with this pair of boxes, moving on to the next redshift pair
                redshifts_LC[i] = z1_LC;
                if(i==0) {
                    start_index_LC[i] = 0;
                    end_index_LC[i] = slice_ct; //This value not inclusive
                }
                else {
                    start_index_LC[i] = end_index_LC[i-1]; // Inclusive of this value
                    end_index_LC[i] = slice_ct;
                }
                z1_LC = z2_LC;
                i += 1;
            }

            total_num_boxes = num_boxes_interp;
            remainder_LC = total_slice_ct - num_boxes_interp*HII_DIM;
        
            final_z = z_LC;
        
            box_z1 = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
            box_z2 = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
            box_interpolate = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
            box_interpolate_remainder = (float *)calloc((float)HII_DIM*(float)HII_DIM*(float)remainder_LC,sizeof(float));
        }
    }
    Stored_LOS_direction_state_2 = LOS_direction;
    
    /////////////////   Read the requisite data if the USE_FCOLL_IONISATION_TABLE option is set for the USE_LIGHTCONE option    /////////////////
    
//    The python script "Create_ionisaton_fcoll_tables.py" which calls the subsequent C file "Createfcoll_ionisation_LC" are used to create this interpolation table.
//     Effectively, this creates an interpolation table for the collapsed fraction for find_HII_bubbles, to avoid having to calculate the full cubic box for each co-eval redshift.
//     This can yield > 20 per cent in computational efficiency at the cost of a small decrease in the accuracy of the neutral fraction, and ionisation fraction per voxel. Check this 
//     accuracy, and whether it is sufficent for your purposes before using this interpolation table.
    
    if(USE_LIGHTCONE) {
        
        if(USE_FCOLL_IONISATION_TABLE) {
            // Note the below code does not yet work for ALPHA != 0
            ReadFcollTable();
            
        }
    }
    
    if((R_BUBBLE_MAX > R_MFP_UB)&&(USE_FCOLL_IONISATION_TABLE==1)) {
        printf("\n");
        printf("The interpolation table for the ionisation box collapse fraction does not have the requisite sampling of the R_MFP\n");
        printf("(The selected R_MFP exceeds the upper bound on R_MFP set in the interpolation table)\n");
        printf("(Either reduce R_MFP or re-run the creation of the interpolation table with a sufficiently large upper bound)\n");
        printf("\n");
        printf("Exiting...");
        printf("\n");
        // Probably should do free the memory properly here...
        return 0;
    }
    
    // Allocate memory for the IGM spin temperature and electron fraction which is stored globally to be taken from Ts.c and used with find_HII_bubbles.c
    Ts_z = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    x_e_z = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    
    // Setup an interpolation table for the error function, helpful for calcluating the collapsed fraction (only for the default model, i.e. mass-independent ionising efficiency)
    erfc_arg_min = -15.0;
    erfc_arg_max = 15.0;
    
    ERFC_NUM_POINTS = 10000;
    
    ERFC_VALS = (double*) calloc(ERFC_NUM_POINTS,sizeof(double));
    ERFC_VALS_DIFF = (double*) calloc(ERFC_NUM_POINTS,sizeof(double));
    
    ArgBinWidth = (erfc_arg_max - erfc_arg_min)/((double)ERFC_NUM_POINTS - 1.);
    InvArgBinWidth = 1./ArgBinWidth;
    
    for(i=0;i<ERFC_NUM_POINTS;i++) {
        
        erfc_arg_val = erfc_arg_min + ArgBinWidth*(double)i;
        
        ERFC_VALS[i] = splined_erfc(erfc_arg_val);
    }
    
    for(i=0;i<(ERFC_NUM_POINTS-1);i++) {
        ERFC_VALS_DIFF[i] = ERFC_VALS[i+1] - ERFC_VALS[i];
    }
    
    // Allocate memory for storing the global history of the IGM neutral fraction and brightness temperature contrast
    if(STORE_DATA) {
        aveNF = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveTb = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveJ_21_LW = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveJ_21_LW_MINI = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveJ_alpha = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveJ_alpha_MINI = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveXheat = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveXheat_MINI = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveNion = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
        aveNion_MINI = (double*) calloc(N_USER_REDSHIFT,sizeof(double));
    }
    
    // if GenerateNewICs == 1, generate the new initial conditions. This calculates the initial conditions in fourier space, and stores the relevant boxes in memory only (nothing is written to file)
    // At the same time, calculate the density field for calculating the IGM spin temperature.
    // This option must be set if the cosmology is to be varied.
    if(GenerateNewICs) {
        HIRES_density = (float *) fftwf_malloc(sizeof(float)*TOT_FFT_NUM_PIXELS);
    
        ComputeInitialConditions();
    
        LOWRES_density_REDSHIFT = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        LOWRES_velocity_REDSHIFT = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        
    }
    
    if (INHOMO_RECO) {
        init_MHR();
    }
    
    // ALLOCATE AND INITIALIZE ADDITIONAL BOXES NEEDED TO KEEP TRACK OF RECOMBINATIONS (Sobacchi & Mesinger 2014; NEW IN v1.3)
    if (INHOMO_RECO){ //  flag in ANAL_PARAMS.H to determine whether to compute recombinations or not
        z_re = (float *) fftwf_malloc(sizeof(float)*HII_TOT_NUM_PIXELS); // the redshift at which the cell is ionized
        Gamma12 = (float *) fftwf_malloc(sizeof(float)*HII_TOT_NUM_PIXELS);  // stores the ionizing backgroud
        N_rec_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS); // cumulative number of recombinations
        N_rec_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
#ifdef MINI_HALO
        J_21_LW = (float *) fftwf_malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        prev_J_21_LW = (float *) fftwf_malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        log10_Mcrit_LW_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        log10_Mcrit_LW_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
#endif 
        
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++) {
            Gamma12[ct] = 0.0;
            z_re[ct] = -1.0;
#ifdef MINI_HALO
            prev_J_21_LW[ct] = 0.0;
#endif
        }
        
        // initialize N_rec
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)N_rec_unfiltered + HII_R_FFT_INDEX(i,j,k)) = 0.0;
                }
            }
        }
    } //  end if INHOMO_RECO
    
#ifdef MINI_HALO
    // this is basically what we do in ComputeIonisationBoxes, we need this to determine how large
    // deltax_prev_filtered shoule be, which will store the filtered deltax in the previous snapshot
    // at all filtering scales... and also prev_Fcoll, and prev_Fcoll_MINI. They all need to be initialized!.
    R=fmax(R_BUBBLE_MIN, (L_FACTOR*BOX_LEN/(float)HII_DIM));
    
    int N_RSTEPS, counter_R, ii, x, y, z;
    
    counter = 0;
    while ((R - fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN)) <= FRACT_FLOAT_ERR ) {
        R*= DELTA_R_HII_FACTOR;
        counter += 1;
    }

    N_RSTEPS = counter + 1;
    deltax_prev_filtered = (fftwf_complex**)fftwf_malloc(N_RSTEPS*sizeof(fftwf_complex *));
    prev_Fcoll           = (float**) calloc(N_RSTEPS,sizeof(float *));
    prev_Fcoll_MINI      = (float**) calloc(N_RSTEPS,sizeof(float *));
    prev_overdense_small_min = (float *) calloc(N_RSTEPS,sizeof(float));
    prev_overdense_large_min = (float *) calloc(N_RSTEPS,sizeof(float));
    prev_overdense_small_bin_width_inv = (float *) calloc(N_RSTEPS,sizeof(float));
    prev_overdense_large_bin_width_inv = (float *) calloc(N_RSTEPS,sizeof(float));

    prev_log10_Fcoll_spline_SFR = (double **) calloc(N_RSTEPS,sizeof(double *));
    prev_Fcoll_spline_SFR = (float **) calloc(N_RSTEPS,sizeof(float *));
    prev_log10_Fcoll_spline_SFR_MINI = (double **) calloc(N_RSTEPS,sizeof(double *));
    prev_Fcoll_spline_SFR_MINI = (float **) calloc(N_RSTEPS,sizeof(float *));

    for (ii=0; ii<N_RSTEPS;ii++){
        deltax_prev_filtered[ii] = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        prev_Fcoll[ii]           = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
        prev_Fcoll_MINI[ii]      = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        prev_log10_Fcoll_spline_SFR[ii] = (double*) calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
        prev_Fcoll_spline_SFR[ii] = (float*) calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));
        prev_log10_Fcoll_spline_SFR_MINI[ii] = (double*) calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
        prev_Fcoll_spline_SFR_MINI[ii] = (float*) calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));

        for (x=0; x<HII_DIM; x++){
            for (y=0; y<HII_DIM; y++){
                for (z=0; z<HII_DIM; z++){
                    *((float *)deltax_prev_filtered[ii] + HII_R_FFT_INDEX(x,y,z)) = -1.5;
                    prev_Fcoll[ii][HII_R_INDEX(x,y,z)] = 0.0;
                    prev_Fcoll_MINI[ii][HII_R_INDEX(x,y,z)] = 0.0;
                }
            }
        }
    }

    //to record the evolution of log10_Mmin_ave for LF calculation
    log10_Mmin_ave = (double*) calloc(N_USER_REDSHIFT, sizeof(double));
    log10_Mmin_MINI_ave = (double*) calloc(N_USER_REDSHIFT, sizeof(double));
    log10_Mmin_ave_spline_acc = gsl_interp_accel_alloc();
    log10_Mmin_ave_spline = gsl_spline_alloc(gsl_interp_cspline, N_USER_REDSHIFT);
    log10_Mmin_MINI_ave_spline_acc = gsl_interp_accel_alloc();
    log10_Mmin_MINI_ave_spline = gsl_spline_alloc(gsl_interp_cspline, N_USER_REDSHIFT);
#endif
        
    /////////////////   Calculate the filtering scales for all the relevant smoothing scales for the HII_BUBBLES excursion set formalism    /////////////////

    ///////////////////////////////// Decide whether or not the spin temperature fluctuations are to be computed (Ts.c) /////////////////////////////////
    if(USE_TS_FLUCT) {
        ///////////////////////////////// Perform 'Ts.c' /////////////////////////////////

        // This will perform the Ts.c part of the computation, and at the end of each redshift sampling, it will calculate the ionisation and brightness temperature boxes.
        ComputeTsBoxes();
    }
    else {
        
        // If here, the spin temperature is assumed to be saturated, and thus only the HII_BUBBLES part of the code is performed.
        // User must provide the co-eval redshifts in this case.
        
        // Reversed the order for iterating to facilitate the code for generating the LC. Here, doesn't matter as assuming Ts >> Tcmb
        for(i=N_USER_REDSHIFT;i--;) {
            // Note, that INHOMO_RECO cannot be set when entering here.
            // INHOMO_RECO must be set with USE_TS_FLUCT
            // This is because no data is stored
            nf_ave = ComputeIonisationBoxes(i,redshifts[i],redshifts[i]+0.2);
        }
    }

    // Storing the global history of the IGM neutral fraction and brightness temperature contrast into a text-file
    if(STORE_DATA) {
        sprintf(filename, "AveData_%f_%f.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
        F=fopen(filename, "wt");
        for(i=0;i<N_USER_REDSHIFT;i++) {
#ifdef MINI_HALO
            fprintf(F,"%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",redshifts[i],aveNF[i],aveTb[i],aveJ_21_LW[i],aveJ_21_LW_MINI[i],aveJ_alpha[i],aveJ_alpha_MINI[i],aveXheat[i],aveXheat_MINI[i],aveNion[i],aveNion_MINI[i]);
#else
            fprintf(F,"%e\t%e\t%e\n",redshifts[i],aveNF[i],aveTb[i]);
#endif
        }
        fclose(F);
    }
    
    // Output the text-file containing the file names of all the 21cm PS calculated from the light-cone boxes
    
    if(USE_LIGHTCONE) {
        
        sprintf(filename, "delTps_lightcone_filenames_%f_%f.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
        F=fopen(filename, "w");
        for(i=0;i<total_num_boxes;i++) {
            fprintf(F,"%s\n",lightcone_box_names[i]);
        }
        fclose(F);
    }
    
#ifdef MINI_HALO
    if (USE_LF){
        // so that redshifts is increasing... :)
        for (i=0; i<N_USER_REDSHIFT; i++){
            redshifts[i] *= -1.;
        }
        gsl_spline_init(log10_Mmin_ave_spline, redshifts, log10_Mmin_ave, N_USER_REDSHIFT);
        gsl_spline_init(log10_Mmin_MINI_ave_spline, redshifts, log10_Mmin_MINI_ave, N_USER_REDSHIFT);
        init_LF_arrays();
        ComputeLF();
        destroy_LF_arrays();
    }
#endif
    
    // De-allocate all arrays etc. that have been allocated and used
    free(Ts_z);
    free(x_e_z);
    
    free(PARAM_VALS);
    
    free(ERFC_VALS);
    free(ERFC_VALS_DIFF);
    
    if(STORE_DATA) {
        free(aveNF);
        free(aveTb);
        free(aveJ_21_LW);
        free(aveJ_21_LW_MINI);
        free(aveJ_alpha);
        free(aveJ_alpha_MINI);
        free(aveXheat);
        free(aveXheat_MINI);
        free(aveNion);
        free(aveNion_MINI);
    }
    free(redshifts);
    
    free(full_index_LC);
    free(slice_redshifts);
    
    if(USE_LIGHTCONE) {
        free(box_z1);
        free(box_z2);
        free(box_interpolate);
        free(box_interpolate_remainder);
        free(redshifts_LC);
        free(start_index_LC);
        free(end_index_LC);
    }
    
    free_ps();

    if (INHOMO_RECO) {
        free_MHR();
    }
    fftwf_free(N_rec_unfiltered);
    fftwf_free(N_rec_filtered);
    fftwf_free(z_re);
    fftwf_free(Gamma12);
#ifdef MINI_HALO
    fftwf_free(log10_Mcrit_LW_unfiltered); 
    fftwf_free(log10_Mcrit_LW_filtered);
    fftwf_free(J_21_LW);
    fftwf_free(prev_J_21_LW);
    for (ii=0; ii<N_RSTEPS;ii++){
        fftwf_free(deltax_prev_filtered[ii]);
        free(prev_Fcoll[ii]);
        free(prev_Fcoll_MINI[ii]);
        free(prev_log10_Fcoll_spline_SFR[ii]);
        free(prev_Fcoll_spline_SFR[ii]);
        free(prev_log10_Fcoll_spline_SFR_MINI[ii]);
        free(prev_Fcoll_spline_SFR_MINI[ii]);
    }
    fftwf_free(deltax_prev_filtered);
    free(prev_Fcoll);
    free(prev_Fcoll_MINI);
    free(prev_log10_Fcoll_spline_SFR);
    free(prev_Fcoll_spline_SFR);
    free(prev_log10_Fcoll_spline_SFR_MINI);
    free(prev_Fcoll_spline_SFR_MINI);
    free(prev_overdense_small_min);
    free(prev_overdense_large_min);
    free(prev_overdense_small_bin_width_inv);
    free(prev_overdense_large_bin_width_inv);
    free(log10_Mmin_ave);
    free(log10_Mmin_MINI_ave);
    gsl_interp_accel_free(log10_Mmin_ave_spline_acc);
    gsl_interp_accel_free(log10_Mmin_MINI_ave_spline_acc);
    gsl_spline_free(log10_Mmin_ave_spline);
    gsl_spline_free(log10_Mmin_MINI_ave_spline);
#endif
    
//    printf("END, time=%06.2f min\n", (double)clock()/CLOCKS_PER_SEC/60.0);

    return 0;
}

void ComputeLF()
{
    char filename[500];
    FILE *F, *OUT;
    int i,i_z;
    double  dlnMhalo, lnMhalo_i, SFRparam, Muv_1, Muv_2, dMuvdMhalo;
    double Luv_over_SFR = 1./1.15/1e-28, delta_lnMhalo = 5e-6;
    /*  
     Luv/SFR = 1 / 1.15 x 10^-28 [M_solar yr^-1/erg s^-1 Hz^-1]
              G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33
    */
    double Mhalo_min = 1e6, Mhalo_max = 1e16;
    double Mhalo_i, lnMhalo_min, lnMhalo_max, lnMhalo_lo, lnMhalo_hi, dlnM;
    float Mlim_Fstar,Fstar;
#ifdef MINI_HALO
    double Mcrit_atom, Mmin_ave ,Mmin_MINI_ave;
    double SFRparam_MINI, Muv_1_MINI, Muv_2_MINI, dMuvdMhalo_MINI;
    float Mlim_Fstar_MINI,Fstar_MINI;
#endif
    // At the moment I just put the redshift list by hand, but this part should be modified.
    float z_LF[NUM_OF_REDSHIFT_FOR_LF] = {6.00, 7.00, 8.00, 10.00, 13.00, 15.00};

    Mlim_Fstar = Mass_limit_bisection((float)Mhalo_min*0.999, (float)Mhalo_max*1.001, ALPHA_STAR, F_STAR10);
#ifdef MINI_HALO
    Mlim_Fstar_MINI = Mass_limit_bisection((float)Mhalo_min*0.999, (float)Mhalo_max*1.001, ALPHA_STAR, F_STAR10_MINI);
#endif

    lnMhalo_min = log(Mhalo_min*0.999);
    lnMhalo_max = log(Mhalo_max*1.001);
    dlnMhalo = (lnMhalo_max - lnMhalo_min)/(double)(NBINS_LF - 1);

    //printf("Calculating LF...\n");
    for (i_z=0; i_z<NUM_OF_REDSHIFT_FOR_LF; i_z++) {
#ifdef MINI_HALO
        Mcrit_atom = atomic_cooling_threshold(z_LF[i_z]);
        Mmin_ave = pow(10., gsl_spline_eval(log10_Mmin_ave_spline, 0.-z_LF[i_z], log10_Mmin_ave_spline_acc));
        Mmin_MINI_ave = pow(10., gsl_spline_eval(log10_Mmin_MINI_ave_spline, 0.-z_LF[i_z], log10_Mmin_MINI_ave_spline_acc));
        //printf("z=%.1f, Mcrit_atom=%5.2e, Mmin_ave=%5.2e, Mmin_MINI_ave=%5.2e\n", z_LF[i_z], Mcrit_atom, Mmin_ave, Mmin_MINI_ave);
#endif

        for (i=0; i<NBINS_LF; i++) {
            // generate interpolation arrays
            lnMhalo_param[i] = lnMhalo_min + dlnMhalo*(double)i;
            Mhalo_i = exp(lnMhalo_param[i]);

            Fstar = F_STAR10*pow(Mhalo_i/1e10,ALPHA_STAR);
            if (Fstar > 1.) Fstar = 1;
#ifdef MINI_HALO
            Fstar_MINI = F_STAR10_MINI*pow(Mhalo_i/1e10,ALPHA_STAR);
            if (Fstar_MINI > 1.) Fstar_MINI = 1;
#endif

            // parametrization of SFR
            SFRparam = Mhalo_i * OMb/OMm * (double)Fstar * (double)(hubble(z_LF[i_z])*SperYR/t_STAR); // units of M_solar/year 

            Muv_param[i] = 51.63 - 2.5*log10(SFRparam*Luv_over_SFR); // UV magnitude

            // except if Muv value is nan or inf, but avoid error put the value as 10.
            if ( isinf(Muv_param[i]) || isnan(Muv_param[i]) ) Muv_param[i] = 10.;
#ifdef MINI_HALO
            SFRparam_MINI = Mhalo_i * OMb/OMm * (double)Fstar_MINI * (double)(hubble(z_LF[i_z])*SperYR/t_STAR); // units of M_solar/year 


            Muv_param_MINI[i] = 51.63 - 2.5*log10(SFRparam_MINI*Luv_over_SFR); // UV magnitude

            // except if Muv value is nan or inf, but avoid error put the value as 10.
            if ( isinf(Muv_param_MINI[i]) || isnan(Muv_param_MINI[i]) ) Muv_param_MINI[i] = 10.;
#endif
        }

        gsl_spline_init(LF_spline, lnMhalo_param, Muv_param, NBINS_LF);
#ifdef MINI_HALO
        gsl_spline_init(LF_spline_MINI, lnMhalo_param, Muv_param_MINI, NBINS_LF);
#endif

        lnMhalo_lo = log(Mhalo_min);
        lnMhalo_hi = log(Mhalo_max);
        dlnM = (lnMhalo_hi - lnMhalo_lo)/(double)(NBINS_LF - 1);

        for (i=0; i<NBINS_LF; i++) {
            // calculate luminosity function
            lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
            Mhalo_param[i] = exp(lnMhalo_i);
        
            Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
            Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

            dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));

            log10phi[i] = log10( dNdM_st(z_LF[i_z],exp(lnMhalo_i)) * exp(-(Mmin_ave/Mhalo_param[i])) / fabs(dMuvdMhalo) );
            if (isinf(log10phi[i]) || isnan(log10phi[i]) || log10phi[i] < -30.) log10phi[i] = -30.;
#ifdef MINI_HALO
            Muv_1_MINI = gsl_spline_eval(LF_spline_MINI, lnMhalo_i - delta_lnMhalo, LF_spline_acc_MINI);
            Muv_2_MINI = gsl_spline_eval(LF_spline_MINI, lnMhalo_i + delta_lnMhalo, LF_spline_acc_MINI);
            dMuvdMhalo_MINI = (Muv_2_MINI - Muv_1_MINI) / (2.*delta_lnMhalo * exp(lnMhalo_i));
            log10phi_MINI[i] = log10( dNdM_st(z_LF[i_z],exp(lnMhalo_i)) * exp(-(Mmin_ave/Mhalo_param[i])) * exp(-(Mhalo_param[i]/Mcrit_atom)) / fabs(dMuvdMhalo_MINI) );
            if (isinf(log10phi_MINI[i]) || isnan(log10phi_MINI[i]) || log10phi_MINI[i] < -40.) log10phi_MINI[i] = -40.;
#endif
        }
#ifdef MINI_HALO
        for (i=0; i<NBINS_LF; i++) {
            Muv_param_MINI[i] *= -1.;
        }
        gsl_spline_init(LF2_spline_MINI, Muv_param_MINI, log10phi_MINI, NBINS_LF);
        gsl_spline_init(LF3_spline_MINI, Muv_param_MINI, Mhalo_param, NBINS_LF);
        for (i=0; i<NBINS_LF; i++) {
            //  -Muv_param_MINI[0] and -Muv_param_MINI[NBINS_LF-1] should be the faintest and brightest, halos outside this range are considered being dominated by atomic cooling
            if ((Muv_param[i] <= -Muv_param_MINI[0]) && (Muv_param[i] >= -Muv_param_MINI[NBINS_LF-1])){
                log10phi_MINI[i] = gsl_spline_eval(LF2_spline_MINI, -Muv_param[i], LF2_spline_acc_MINI);
                Mhalo_param_MINI[i] = gsl_spline_eval(LF3_spline_MINI, -Muv_param[i], LF3_spline_acc_MINI);
            }
            else{
                log10phi_MINI[i] = -40;
                if (Muv_param[i] > -Muv_param_MINI[0])
                    Mhalo_param_MINI[i] = Mhalo_min;
                else
                    Mhalo_param_MINI[i] = Mhalo_max;
            }
        }
#endif
        if(PRINT_FILES) {
            sprintf(filename, "LF_estimate_%f_%f_%.6f.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2,z_LF[i_z]);
            F=fopen(filename, "wt");
            for (i=0; i<NBINS_LF; i++){
#ifdef MINI_HALO
                fprintf(F, "%e\t%e\t%e\t%e\t%e\t%e\n", Muv_param[i],log10(pow(10.,log10phi[i]) + pow(10.,log10phi_MINI[i])),log10phi[i],Mhalo_param[i],log10phi_MINI[i],Mhalo_param_MINI[i]);
#else
                fprintf(F, "%e\t%e\t%e\n", Muv_param[i],log10phi[i],Mhalo_param[i]);
#endif
            }
            fclose(F);
        }
    }
}

void ComputeTsBoxes() {
    
    /* This is an entire re-write of Ts.c from 21cmFAST. You can refer back to Ts.c in 21cmFAST if this become a little obtuse. The computation has remained the same */
    
    /////////////////// Defining variables for the computation of Ts.c //////////////
    char filename[500];
    char wisdom_filename[500];
    FILE *F, *OUT;
    fftwf_plan plan;
    
    unsigned long long ct, FCOLL_SHORT_FACTOR;
    
    int R_ct,i,ii,j,k,i_z,COMPUTE_Ts,x_e_ct,m_xHII_low,m_xHII_high,n_ct, zpp_gridpoint1_int, zpp_gridpoint2_int,zpp_evolve_gridpoint1_int, zpp_evolve_gridpoint2_int;
    
    short dens_grid_int;
    
    double Tk_ave, J_alpha_ave, xalpha_ave, J_alpha_tot, Xheat_ave, Xion_ave, nuprime, Ts_ave, lower_int_limit,Luminosity_converstion_factor,T_inv_TS_fast_inv, nfgave, nf_ave;
#ifdef MINI_HALO
    double J_LW_ave, Luminosity_converstion_factor_MINI, J_alpha_tot_MINI, J_alpha_ave_MINI, J_LW_ave_MINI,dxheat_dzp_MINI,Xheat_ave_MINI;
#endif
    double dadia_dzp, dcomp_dzp, dxheat_dt, dxion_source_dt, dxion_sink_dt, T, x_e, dxe_dzp, n_b, dspec_dzp, dxheat_dzp, dxlya_dt, dstarlya_dt, fcoll_R;
    double Trad_fast,xc_fast,xc_inverse,TS_fast,TSold_fast,xa_tilde_fast,TS_prefactor,xa_tilde_prefactor,T_inv,T_inv_sq,xi_power,xa_tilde_fast_arg,Trad_fast_inv,TS_fast_inv,dcomp_dzp_prefactor;

    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, mu_for_Ts, filling_factor_of_HI_zp, dzp, prev_zp, zpp, prev_zpp, prev_R, Tk_BC, xe_BC;
    float xHII_call, curr_xalpha, TK, TS, xe, deltax_highz;
    float zpp_for_evolve,dzpp_for_evolve;

    float zpp_grid, zpp_gridpoint1, zpp_gridpoint2,zpp_evolve_gridpoint1, zpp_evolve_gridpoint2, grad1, grad2, grad3, grad4, delNL0_bw_val;
    float OffsetValue, DensityValueLow, min_density, max_density;
    
    double curr_delNL0, inverse_val,prefactor_1,prefactor_2,dfcoll_dz_val, density_eval1, density_eval2, grid_sigmaTmin, grid_dens_val, dens_grad, dens_width;
#ifdef MINI_HALO
    double prefactor_2_MINI, dfcoll_dz_val_MINI;
#endif

    float M_MIN_WDM =  M_J_WDM();
    
    double ave_fcoll, ave_fcoll_inv, dfcoll_dz_val_ave;
#ifdef MINI_HALO
    double ave_fcoll_MINI, ave_fcoll_inv_MINI, dfcoll_dz_val_ave_MINI;
#endif
    double total_time, total_time2, total_time3, total_time4;
    
    float curr_dens, min_curr_dens, max_curr_dens;
    
    min_curr_dens = max_curr_dens = 0.;
    
    int fcoll_int_min, fcoll_int_max;
    
    fcoll_int_min = fcoll_int_max = 0;
    
    clock_t start_section, end_section, start_single_z, end_single_z, start_subsection, end_subsection;
    
    float fcoll_interp_min, fcoll_interp_bin_width, fcoll_interp_bin_width_inv, fcoll_interp_val1, fcoll_interp_val2, dens_val;
    float fcoll_interp_high_min, fcoll_interp_high_bin_width, fcoll_interp_high_bin_width_inv;
    int dens_int;
    float dens_diff;
    
    float ave_fcoll_float, dfcoll_dz_val_float, fcoll_float;
    
    float redshift_table_fcollz_diff,redshift_table_fcollz_diff_Xray;
    
    int redshift_int_fcollz,redshift_int_fcollz_Xray;

#ifdef MINI_HALO
    float log10_Mcrit_LW_val, log10_Mcrit_LW_diff, log10_Mcrit_LW_ave_table_fcollz;
    int log10_Mcrit_LW_int, log10_Mcrit_LW_ave_int_fcollz;
    float *log10_Mcrit_LW[NUM_FILTER_STEPS_FOR_Ts];
#endif
    
    
    float ln_10;
    
    ln_10 = logf(10);
    
    
    int Tvir_min_int,Numzp_for_table,counter;
    double X_RAY_Tvir_BinWidth;
    // New in v1.4
    int arr_num;
    float zp_table, zpp_integrand;
    float fcoll;
    float Splined_Fcoll,Splined_Fcollzp_mean,Splined_Fcollzpp_X_mean;
    // New in v1.5
#ifdef MINI_HALO
    int index_left, index_right;
    float zpp_integrand_MINI;

    float fcoll_MINI_left, fcoll_MINI_right, fcoll_MINI;
    float Splined_Fcoll_MINI,Splined_Fcollzp_mean_MINI_left, Splined_Fcollzp_mean_MINI_right, Splined_Fcollzp_mean_MINI;
    float Splined_Fcollzpp_X_mean_MINI_left, Splined_Fcollzpp_X_mean_MINI_right, Splined_Fcollzpp_X_mean_MINI;
#endif

    
    //printf("F_STAR10 = %.4f, ALPHA_STAR = %.4f, F_ESC10 = %.4f, ALPHA_ESC = %.4e, M_MIN = %.4e, M_TURN = %.4e\n",F_STAR10, ALPHA_STAR, F_ESC10, ALPHA_ESC, M_MIN, M_TURN); // TEST
    
    X_RAY_Tvir_BinWidth = (X_RAY_Tvir_UPPERBOUND - X_RAY_Tvir_LOWERBOUND)/( (double)X_RAY_Tvir_POINTS - 1. );
    
    // Can speed up computation (~20%) by pre-sampling the fcoll field as a function of X_RAY_TVIR_MIN (performed by calling CreateFcollTable.
    // Can be helpful when HII_DIM > ~128, otherwise its easier to just do the full box
    // This table can be created using "CreateFcollTable.c". See this file for further details.
    if(SHORTEN_FCOLL) {
        
        Tvir_min_int = (int)floor( (log10(X_RAY_Tvir_MIN) - X_RAY_Tvir_LOWERBOUND)/X_RAY_Tvir_BinWidth );
        
        sprintf(filename, "FcollTvirTable_Numzp_ZPRIME_FACTOR%0.2f_logTvirmin%0.6f_logTvirmax%0.6f_XRAY_POINTS%d_z_end%06.6f_%0.2fMpc_%d.dat",ZPRIME_STEP_FACTOR,X_RAY_Tvir_LOWERBOUND,X_RAY_Tvir_UPPERBOUND,X_RAY_Tvir_POINTS,REDSHIFT,BOX_LEN,HII_DIM);
        F = fopen(filename, "rb");
        fread(&Numzp_for_table, sizeof(int),1,F);
        fclose(F);
    }
    else {
        // Need to take on some number for the memory allocation
        Numzp_for_table = 1;
    }
    
    float *min_densities = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    float *max_densities = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    
    // Allocate the memory for this interpolation table
    double ***Fcoll_R_Table = (double ***)calloc(Numzp_for_table,sizeof(double **));
    for(i=0;i<Numzp_for_table;i++) {
        Fcoll_R_Table[i] = (double **)calloc(X_RAY_Tvir_POINTS,sizeof(double *));
        for(j=0;j<X_RAY_Tvir_POINTS;j++) {
            Fcoll_R_Table[i][j] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
    }
#ifdef MINI_HALO
    prev_mean_f_coll_st = 0.0;
    prev_mean_f_coll_st_MINI = 0.0;
#endif
    
    if(SHORTEN_FCOLL) {
        
        sprintf(filename, "FcollTvirTable_ZPRIME_FACTOR%0.2f_logTvirmin%0.6f_logTvirmax%0.6f_XRAY_POINTS%d_z_end%06.6f_%0.2fMpc_%d.dat",ZPRIME_STEP_FACTOR,X_RAY_Tvir_LOWERBOUND,X_RAY_Tvir_UPPERBOUND,X_RAY_Tvir_POINTS,REDSHIFT,BOX_LEN,HII_DIM);
        F = fopen(filename, "rb");
        for(i=0;i<Numzp_for_table;i++) {
            for(j=0;j<X_RAY_Tvir_POINTS;j++) {
                fread(Fcoll_R_Table[i][j], sizeof(double),NUM_FILTER_STEPS_FOR_Ts,F);
            }
        }
    }
    
    // Initialise arrays to be used for the Ts.c computation //
    init_21cmMC_Ts_arrays();
#ifdef MINI_HALO
    for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        log10_Mcrit_LW[R_ct] = (float *) calloc(HII_TOT_NUM_PIXELS, sizeof(float));
    }
#endif
    
    ///////////////////////////////  BEGIN INITIALIZATION   //////////////////////////////
    growth_factor_z = dicke(REDSHIFT);
    inverse_growth_factor_z = 1./growth_factor_z;
    
    /*if (X_RAY_Tvir_MIN < 9.99999e3) // neutral IGM
        mu_for_Ts = 1.22;
    else // ionized IGM
        mu_for_Ts = 0.6;*/
    
    //set the minimum ionizing source mass
    // In v1.4 the miinimum ionizing source mass does not depend on redshift.
    // For the constant ionizing efficiency parameter, M_MIN is set to be M_TURN which is a sharp cut-off.
    // For the new parametrization, the number of halos hosting active galaxies (i.e. the duty cycle) is assumed to
    // exponentially decrease below M_TURNOVER Msun, : fduty \propto e^(- M_TURNOVER / M)
    // In this case, we define M_MIN = M_TURN/50, i.e. the M_MIN is integration limit to compute follapse fraction.
    //M_MIN_at_z = get_M_min_ion(REDSHIFT);
    if (!USE_MASS_DEPENDENT_ZETA) M_MIN = M_TURN;
    
    // Initialize some interpolation tables
    init_heat();
    
    // check if we are in the really high z regime before the first stars; if so, simple
    if (REDSHIFT > Z_HEAT_MAX){
    
        /**** NOTE: THIS NEEDS TO CHANGE. Though it'll only cause problems if this condition is met (which should never happen) ****/

        /**** NOTE from YQ: I don't modify this part, because "it should never happen?" ****/
        
        xe = xion_RECFAST(REDSHIFT,0);
        TK = T_RECFAST(REDSHIFT,0);
        
        // open input
        sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc",REDSHIFT, HII_DIM, BOX_LEN);
        F = fopen(filename, "rb");
        
        // open output
        // New in v1.4
        if (USE_MASS_DEPENDENT_ZETA) {
#ifdef MINI_HALO
        sprintf(filename, "../Boxes/Ts_z%06.2f_L_X%.1e_alphaX%.1f_f_star%06.4f_L_X_MINI%.1e_alphaX_MINI%.1f_f_star_MINI%06.4f_alpha_star%06.4f_t_star%06.4f_%i_%.0fMpc", REDSHIFT, L_X, X_RAY_SPEC_INDEX, F_STAR10, L_X_MINI, X_RAY_SPEC_INDEX_MINI, F_STAR10_MINI, ALPHA_STAR, t_STAR, HII_DIM, BOX_LEN);
#else
        sprintf(filename, "../Boxes/Ts_z%06.2f_L_X%.1e_alphaX%.1f_f_star%06.4f_alpha_star%06.4f_MturnX%.1e_t_star%06.4f_Pop%i_%i_%.0fMpc", REDSHIFT, L_X, X_RAY_SPEC_INDEX, F_STAR10, ALPHA_STAR, M_TURN, t_STAR, Pop, HII_DIM, BOX_LEN);
#endif
        }
        else {
        sprintf(filename, "../Boxes/Ts_z%06.2f_L_X%.1e_alphaX%.1f_MminX%.1e_zetaIon%.2f_Pop%i_%i_%.0fMpc", REDSHIFT, L_X, X_RAY_SPEC_INDEX, M_MIN, HII_EFF_FACTOR, Pop, HII_DIM, BOX_LEN);
        }
        
        // read file
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    fread(&deltax_highz, sizeof(float), 1, F);
                    
                    // compute the spin temperature
                    TS = get_Ts(REDSHIFT, deltax_highz, TK, xe, 0, &curr_xalpha);
                    
                    // and print it out
                    fwrite(&TS, sizeof(float), 1, OUT);
                }
            }
        }
        
        destruct_heat(); fclose(F); fclose(OUT);
    }
    else {
        
        // set boundary conditions for the evolution equations->  values of Tk and x_e at Z_HEAT_MAX
        if (XION_at_Z_HEAT_MAX > 0) // user has opted to use his/her own value
            xe_BC = XION_at_Z_HEAT_MAX;
        else// will use the results obtained from recfast
            xe_BC = xion_RECFAST(Z_HEAT_MAX,0);
        if (TK_at_Z_HEAT_MAX > 0)
            Tk_BC = TK_at_Z_HEAT_MAX;
        else
            Tk_BC = T_RECFAST(Z_HEAT_MAX,0);
        
        /////////////// Create the z=0 non-linear density fields smoothed on scale R to be used in computing fcoll //////////////
        R = L_FACTOR*BOX_LEN/(float)HII_DIM;
        R_factor = pow(R_XLy_MAX/R, 1./(float)NUM_FILTER_STEPS_FOR_Ts);
//      R_factor = pow(E, log(HII_DIM)/(float)NUM_FILTER_STEPS_FOR_Ts);
        
        ///////////////////  Read in density box at z-prime  ///////////////
        if(GenerateNewICs) {
        
            // If GenerateNewICs == 1, we are generating a new set of initial conditions and density field. Hence, calculate the density field to be used for Ts.c
            
            ComputePerturbField(REDSHIFT);
                
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        *((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k)) = LOWRES_density_REDSHIFT[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }
        else {
            
            // Read in a pre-computed density field which is stored in the "Boxes" folder

            // allocate memory for the nonlinear density field and open file
            sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc",REDSHIFT, HII_DIM, BOX_LEN);
            F = fopen(filename, "rb");
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        fread((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F);
                    }
                }
            }
            fclose(F);
        }
        
        ////////////////// Transform unfiltered box to k-space to prepare for filtering /////////////////
        if(USE_FFTW_WISDOM) {
            // Check to see if the required wisdom exists, and then create it if it doesn't
            sprintf(wisdom_filename,"../FFTW_Wisdoms/real_to_complex_%d.fftwf_wisdom",HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Export the wisdom to filename for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc",REDSHIFT, HII_DIM, BOX_LEN);
                F = fopen(filename, "rb");
                for (i=0; i<HII_DIM; i++){
                    for (j=0; j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            fread((float *)unfiltered_box + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F);
                        }
                    }
                }
                fclose(F);
                
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)unfiltered_box, (fftwf_complex *)unfiltered_box, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }

        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
        }
        
        // Smooth the density field, at the same time store the minimum and maximum densities for their usage in the interpolation tables
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){

            R_values[R_ct] = R;
            // New in v1.4: in the new parametrization, this function is not used to compute collapse fraction.
            if (!USE_MASS_DEPENDENT_ZETA) sigma_atR[R_ct] = sigma_z0(RtoM(R));
            //sigma_atR[R_ct] = sigma_z0(RtoM(R));
        
            // copy over unfiltered box
            memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
            if (R_ct > 0){ // don't filter on cell size
                HII_filter(box, HEAT_FILTER, R);
            }
            // now fft back to real space
            if(USE_FFTW_WISDOM) {
                // Check to see if the wisdom exists, create it if it doesn't
                sprintf(wisdom_filename,"../FFTW_Wisdoms/complex_to_real_%d.fftwf_wisdom",HII_DIM);
                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
                else {
                    if(R_ct==0) {
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_PATIENT);
                        fftwf_execute(plan);
                        
                        // Store the wisdom for later use
                        fftwf_export_wisdom_to_filename(wisdom_filename);
                        
                        // copy over unfiltered box
                        memcpy(box, unfiltered_box, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                        
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                    else {
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                }
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)box, (float *)box, FFTW_ESTIMATE);
                fftwf_execute(plan);
            }
            
            min_density = 0.0;
            max_density = 0.0;

            // copy over the values
            for (i=HII_DIM; i--;){
                for (j=HII_DIM; j--;){
                    for (k=HII_DIM; k--;){
                        curr_delNL0 = *((float *) box + HII_R_FFT_INDEX(i,j,k));
                            
                        if (curr_delNL0 < -1){ // correct for alliasing in the filtering step
                            curr_delNL0 = -1.+FRACT_FLOAT_ERR;
                        }
                            
                        // and linearly extrapolate to z=0
                        curr_delNL0 *= inverse_growth_factor_z;
                        
                        if(USE_MASS_DEPENDENT_ZETA) {
                            delNL0[R_ct][HII_R_INDEX(i,j,k)] = curr_delNL0;
                        }
                        else {
                            delNL0_rev[HII_R_INDEX(i,j,k)][R_ct] = curr_delNL0;
                        }
                        
                        if(curr_delNL0 < min_density) {
                            min_density = curr_delNL0;
                        }
                        if(curr_delNL0 > max_density) {
                            max_density = curr_delNL0;
                        }
                    }
                }
            }
            if(!USE_MASS_DEPENDENT_ZETA) {
                if(min_density < 0.0) {
                    delNL0_LL[R_ct] = min_density*1.001;
                    delNL0_Offset[R_ct] = 1.e-6 - (delNL0_LL[R_ct]);
                }
                else {
                    delNL0_LL[R_ct] = min_density*0.999;
                    delNL0_Offset[R_ct] = 1.e-6 + (delNL0_LL[R_ct]);
                }
                   if(max_density < 0.0) {
                    delNL0_UL[R_ct] = max_density*0.999;
                }
                else {
                    delNL0_UL[R_ct] = max_density*1.001;
                }
            }
            
            min_densities[R_ct] = min_density;
            max_densities[R_ct] = max_density;
    
            R *= R_factor;
                
        } //end for loop through the filter scales R
        
        // and initialize to the boundary values at Z_HEAT_END
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            Tk_box[ct] = Tk_BC;
            x_e_box[ct] = xe_BC;
        }
        x_e_ave = xe_BC;
        Tk_ave = Tk_BC;
    
        ////////////////////////////    END INITIALIZATION   /////////////////////////////
    
        // main trapezoidal integral over z' (see eq. ? in Mesinger et al. 2009)
        zp = REDSHIFT*1.0001; //higher for rounding
        // New in v1.4: count the number of zp steps
        while (zp < Z_HEAT_MAX) {
            zp = ((1.+zp)*ZPRIME_STEP_FACTOR - 1);

        }
        prev_zp = Z_HEAT_MAX;
        zp = ((1.+zp)/ ZPRIME_STEP_FACTOR - 1);
        dzp = zp - prev_zp;
        COMPUTE_Ts = 0;
    
        determine_zpp_min = REDSHIFT*0.999;
    
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            if (R_ct==0){
                prev_zpp = zp;
                prev_R = 0;
            }
            else{
                prev_zpp = zpp_edge[R_ct-1];
                prev_R = R_values[R_ct-1];
            }
            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
        }
    
        determine_zpp_max = zpp*1.001;
    
        ////////////////////////////    Create and fill interpolation tables to be used by Ts.c   /////////////////////////////
        
        // An interpolation table for f_coll (delta vs redshift)
        if (USE_MASS_DEPENDENT_ZETA) {
            zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);

            // generates an interpolation table for redshift
            for (i=0; i<zpp_interp_points_SFR;i++) {
                //zpp_interp_table[i] = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points_SFR-1.0);
                zpp_interp_table[i] = determine_zpp_min + zpp_bin_width*(float)i;
            }

            /* initialise interpolation of the mean collapse fraction for global reionization.*/
#ifdef MINI_HALO
            initialise_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10, F_STAR10_MINI, F_ESC_MINI);
            
#ifndef UES_KERAS
            initialise_Xray_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, ALPHA_STAR, F_STAR10, F_STAR10_MINI);
#endif
#else
            initialise_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, M_TURN, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10);
            
#ifndef UES_KERAS
            initialise_Xray_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, M_TURN, ALPHA_STAR, F_STAR10);
#endif
#endif
            
            zp_table = zp;
            counter = 0;
            for (i=0; i<N_USER_REDSHIFT; i++) {
                  for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                      if (R_ct==0){
                          prev_zpp = zp_table;
                          prev_R = 0;
                      }
                      else{
                          prev_zpp = zpp_edge[R_ct-1];
                          prev_R = R_values[R_ct-1];
                      }
                    zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
                    zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
                    redshift_interp_table[counter] = zpp;
                    growth_interp_table[counter] = dicke(zpp);
#ifdef MINI_HALO
                    Mcrit_atom_interp_table[counter] = atomic_cooling_threshold(zpp);
#endif
                    counter += 1;
                  }
                prev_zp = zp_table;
                zp_table = ((1.+prev_zp) / ZPRIME_STEP_FACTOR - 1);
            }

            /* generate a table for interpolation of the collapse fraction with respect to the X-ray heating, as functions of
            filtering scale, redshift and overdensity.
               Note that at a given zp, zpp values depends on the filtering scale R, i.e. f_coll(z(R),delta).
               Compute the conditional mass function, but assume f_{esc10} = 1 and \alpha_{esc} = 0. */
#ifdef MINI_HALO
            initialise_Xray_Fcollz_SFR_Conditional_table(NUM_FILTER_STEPS_FOR_Ts,min_densities,max_densities,growth_interp_table,R_values, Mcrit_atom_interp_table, ALPHA_STAR, F_STAR10, F_STAR10_MINI);
#else
            initialise_Xray_Fcollz_SFR_Conditional_table(NUM_FILTER_STEPS_FOR_Ts,min_densities,max_densities,growth_interp_table,R_values, M_TURN, ALPHA_STAR, F_STAR10);
#endif

        }
        else {
            init_FcollTable(determine_zpp_min,determine_zpp_max);
            zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points-1.0);
            dens_width = 1./((double)dens_Ninterp - 1.);
        
            // Determine the sampling of the density values, for the various interpolation tables
            for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
                log10delNL0_diff_UL[ii] = log10( delNL0_UL[ii] + delNL0_Offset[ii] );
                log10delNL0_diff[ii] = log10( delNL0_LL[ii] + delNL0_Offset[ii] );
                delNL0_bw[ii] = ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width;
                delNL0_ibw[ii] = 1./delNL0_bw[ii];
            }
        
            // Gridding the density values for the interpolation tables
            for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
                for(j=0;j<dens_Ninterp;j++) {
                    grid_dens[ii][j] = log10delNL0_diff[ii] + ( log10delNL0_diff_UL[ii] - log10delNL0_diff[ii] )*dens_width*(double)j;
                    grid_dens[ii][j] = pow(10,grid_dens[ii][j]) - delNL0_Offset[ii];
                }
            }
        
            // Calculate the sigma_z and Fgtr_M values for each point in the interpolation table
            for(i=0;i<zpp_interp_points;i++) {
                zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points-1.0);
        
                //Sigma_Tmin_grid[i] = sigma_z0(FMAX(TtoM(zpp_grid, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                //ST_over_PS_arg_grid[i] = FgtrM_st(zpp_grid, FMAX(TtoM(zpp_grid, X_RAY_Tvir_MIN, mu_for_Ts),  M_MIN_WDM));
                // New in v1.4: halo mass does not depend on Tvir
                Sigma_Tmin_grid[i] = sigma_z0(M_MIN);
                ST_over_PS_arg_grid[i] = FgtrM_st(zpp_grid, M_MIN);
            }
        
            // Create the interpolation tables for the derivative of the collapsed fraction and the collapse fraction itself
            for(ii=0;ii<NUM_FILTER_STEPS_FOR_Ts;ii++) {
                for(i=0;i<zpp_interp_points;i++) {

                    zpp_grid = determine_zpp_min + (determine_zpp_max - determine_zpp_min)*(float)i/((float)zpp_interp_points-1.0);
                    grid_sigmaTmin = Sigma_Tmin_grid[i];

                    for(j=0;j<dens_Ninterp;j++) {
                
                        grid_dens_val = grid_dens[ii][j];
                        if(!SHORTEN_FCOLL) {
                            fcoll_R_grid[ii][i][j] = sigmaparam_FgtrM_bias(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                        }
                        dfcoll_dz_grid[ii][i][j] = dfcoll_dz(zpp_grid, grid_sigmaTmin, grid_dens_val, sigma_atR[ii]);
                    }
                }
               }

            // Determine the grid point locations for solving the interpolation tables
            for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                    SingleVal_int[R_ct] = (short)floor( ( log10(delNL0_rev[box_ct][R_ct] + delNL0_Offset[R_ct]) - log10delNL0_diff[R_ct] )*delNL0_ibw[R_ct]);
                }
                memcpy(dens_grid_int_vals[box_ct],SingleVal_int,sizeof(short)*NUM_FILTER_STEPS_FOR_Ts);
            }
    
            // Evaluating the interpolated density field points (for using the interpolation tables for fcoll and dfcoll_dz)
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                OffsetValue = delNL0_Offset[R_ct];
                DensityValueLow = delNL0_LL[R_ct];
                delNL0_bw_val = delNL0_bw[R_ct];
            
                for(i=0;i<dens_Ninterp;i++) {
                    density_gridpoints[i][R_ct] = pow(10.,( log10( DensityValueLow + OffsetValue) + delNL0_bw_val*((float)i) )) - OffsetValue;
                }
            }
        }
        
        counter = 0;
        
        // This is the main loop for calculating the IGM spin temperature. Structure drastically different from Ts.c in 21cmFAST, however algorithm and computation remain the same.
        while (zp > REDSHIFT){

            // check if we will next compute the spin temperature (i.e. if this is the final zp step)
            if (Ts_verbose || (((1.+zp) / ZPRIME_STEP_FACTOR) < (REDSHIFT+1)) )
                COMPUTE_Ts = 1;
            
            // check if we are in the really high z regime before the first stars..
            // New in v1.4
            if (USE_MASS_DEPENDENT_ZETA) { // New in v1.4
                
                redshift_int_fcollz = (int)floor( ( zp - determine_zpp_min )/zpp_bin_width );
                
                redshift_table_fcollz_diff = ( zp - determine_zpp_min - zpp_bin_width*(float)redshift_int_fcollz ) / zpp_bin_width;
                
                Splined_Fcollzp_mean = Fcollz_val[redshift_int_fcollz] + redshift_table_fcollz_diff *( Fcollz_val[redshift_int_fcollz+1] - Fcollz_val[redshift_int_fcollz] );
                if (Splined_Fcollzp_mean < 0.) Splined_Fcollzp_mean = 1e-40;
#ifdef MINI_HALO
                log10_Mcrit_mol = log10(lyman_werner_threshold(zp, 0.));
                log10_Mcrit_LW_ave = 0.0;
                for (i=0; i<HII_DIM; i++){
                  for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                      *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k)) = log10(lyman_werner_threshold(zp, prev_J_21_LW[HII_R_INDEX(i,j,k)]));
                      log10_Mcrit_LW_ave += *((float *)log10_Mcrit_LW_unfiltered + HII_R_FFT_INDEX(i,j,k));
                    }
                  }
                }
                log10_Mcrit_LW_ave /= HII_TOT_NUM_PIXELS;

                log10_Mcrit_LW_ave_int_fcollz = (int)floor( ( log10_Mcrit_LW_ave - LOG10MTURN_MIN) / LOG10MTURN_INT);
                log10_Mcrit_LW_ave_table_fcollz = LOG10MTURN_MIN + LOG10MTURN_INT * (float)log10_Mcrit_LW_ave_int_fcollz;
                index_left  = redshift_int_fcollz + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_fcollz;
                index_right = redshift_int_fcollz + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_fcollz+1);

                Splined_Fcollzp_mean_MINI_left = Fcollz_val_MINI[index_left] + redshift_table_fcollz_diff * ( Fcollz_val_MINI[index_left + 1] - Fcollz_val_MINI[index_left] );
                Splined_Fcollzp_mean_MINI_right = Fcollz_val_MINI[index_right] + redshift_table_fcollz_diff * ( Fcollz_val_MINI[index_right + 1] - Fcollz_val_MINI[index_right] );
                Splined_Fcollzp_mean_MINI = Splined_Fcollzp_mean_MINI_left + (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_fcollz) / LOG10MTURN_INT * ( Splined_Fcollzp_mean_MINI_right - Splined_Fcollzp_mean_MINI_left );
                if (Splined_Fcollzp_mean_MINI < 0.) Splined_Fcollzp_mean_MINI = 1e-40;

                // NEED TO FILTER Mcrit_LW!!!
                /*** Transform unfiltered box to k-space to prepare for filtering ***/
                if(USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mcrit_LW_unfiltered, (fftwf_complex *)log10_Mcrit_LW_unfiltered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mcrit_LW_unfiltered, (fftwf_complex *)log10_Mcrit_LW_unfiltered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);
                for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++)
                    log10_Mcrit_LW_unfiltered[ct] /= (float)HII_TOT_NUM_PIXELS;

                if ( Splined_Fcollzp_mean + Splined_Fcollzp_mean_MINI < 1e-15 )
#else
                if ( Splined_Fcollzp_mean < 1e-15 )
#endif
                    NO_LIGHT = 1; 
                  else 
                    NO_LIGHT = 0; 
            }
            else {
                  if (FgtrM(zp, M_MIN) < 1e-15 )
                    NO_LIGHT = 1; 
                  else 
                    NO_LIGHT = 0; 
            }    
            
            // New in v1.4
            if (USE_MASS_DEPENDENT_ZETA) {
#ifdef MINI_HALO
                filling_factor_of_HI_zp = 1 - (ION_EFF_FACTOR * Splined_Fcollzp_mean + ION_EFF_FACTOR_MINI * Splined_Fcollzp_mean_MINI) / (1.0 - x_e_ave);
#else
                filling_factor_of_HI_zp = 1 - ION_EFF_FACTOR * Splined_Fcollzp_mean / (1.0 - x_e_ave);
#endif
            }
            else {
                filling_factor_of_HI_zp = 1 - ION_EFF_FACTOR * FgtrM_st(zp, M_MIN) / (1.0 - x_e_ave);
            }
            if (filling_factor_of_HI_zp > 1) filling_factor_of_HI_zp=1;
            
            // let's initialize an array of redshifts (z'') corresponding to the
            // far edge of the dz'' filtering shells
            // and the corresponding minimum halo scale, sigma_Tmin,
            // as well as an array of the frequency integrals
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                if (R_ct==0){
                    prev_zpp = zp;
                    prev_R = 0;
                }
                else{
                    prev_zpp = zpp_edge[R_ct-1];
                    prev_R = R_values[R_ct-1];
                }
                zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
                zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
                zpp_for_evolve_list[R_ct] = zpp;

                if (R_ct==0){
                    dzpp_for_evolve = zp - zpp_edge[0];
                }
                else{
                    dzpp_for_evolve = zpp_edge[R_ct-1] - zpp_edge[R_ct];
                }
                zpp_growth[R_ct] = dicke(zpp);
                fcoll_R_array[R_ct] = 0.0;

#ifdef MINI_HALO
                memcpy(log10_Mcrit_LW_filtered, log10_Mcrit_LW_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                if (R_ct > 0){// don't filter on cell size
                  HII_filter(log10_Mcrit_LW_filtered, HEAT_FILTER, R_values[R_ct]);
                }
                // now fft back to real space
                if(USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mcrit_LW_filtered, (float *)log10_Mcrit_LW_filtered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mcrit_LW_filtered, (float *)log10_Mcrit_LW_filtered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);

                log10_Mcrit_LW_ave = 0; //recalculate it at this filtering scale
                for (i=0; i<HII_DIM; i++){
                  for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                      log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = *((float *) log10_Mcrit_LW_filtered + HII_R_FFT_INDEX(i,j,k));
                      if(log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] < log10_Mcrit_mol)
                          log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = log10_Mcrit_mol;
                      if (log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] > LOG10MTURN_MAX)
                          log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)] = LOG10MTURN_MAX;
                      log10_Mcrit_LW_ave += log10_Mcrit_LW[R_ct][HII_R_INDEX(i,j,k)];
                    }
                  }
                }
                log10_Mcrit_LW_ave /= HII_TOT_NUM_PIXELS;
                log10_Mcrit_LW_ave_int_fcollz = (int)floor( ( log10_Mcrit_LW_ave - LOG10MTURN_MIN) / LOG10MTURN_INT);
                log10_Mcrit_LW_ave_table_fcollz = LOG10MTURN_MIN + LOG10MTURN_INT * (float)log10_Mcrit_LW_ave_int_fcollz;
#endif

                if (USE_MASS_DEPENDENT_ZETA) { 
                    // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                    
                    redshift_int_fcollz_Xray = (int)floor( ( zpp - determine_zpp_min )/zpp_bin_width );
                    
                    redshift_table_fcollz_diff_Xray = ( zpp - determine_zpp_min - zpp_bin_width*(float)redshift_int_fcollz_Xray ) /zpp_bin_width;
                    
#ifdef USE_KERAS
					Splined_Fcollzpp_X_mean = FcollzX_val_emulator(F_STAR10_norm, ALPHA_STAR_norm, SIGMA8_norm, REDSHIFT_norm);
#else
                    Splined_Fcollzpp_X_mean = FcollzX_val[redshift_int_fcollz_Xray] + redshift_table_fcollz_diff_Xray *( FcollzX_val[redshift_int_fcollz_Xray+1] - FcollzX_val[redshift_int_fcollz_Xray] );
#endif
                    if (Splined_Fcollzpp_X_mean < 0.) Splined_Fcollzpp_X_mean = 1e-40;

                    ST_over_PS[R_ct] = pow(1.+zpp, -X_RAY_SPEC_INDEX)*fabs(dzpp_for_evolve);
                    ST_over_PS[R_ct] *= Splined_Fcollzpp_X_mean;
#ifdef MINI_HALO
                    index_left = redshift_int_fcollz_Xray + zpp_interp_points_SFR * log10_Mcrit_LW_ave_int_fcollz;
                    index_right = redshift_int_fcollz_Xray + zpp_interp_points_SFR * (log10_Mcrit_LW_ave_int_fcollz + 1);
                    // for log10_Mcrit_LW_ave_int_fcollz, we don't know how it evovles, so just use the same...
                    Splined_Fcollzpp_X_mean_MINI_left = FcollzX_val_MINI[index_left] +  redshift_table_fcollz_diff_Xray * ( FcollzX_val_MINI[index_left + 1] - FcollzX_val_MINI[index_left] );
                    Splined_Fcollzpp_X_mean_MINI_right = FcollzX_val_MINI[index_right] + redshift_table_fcollz_diff_Xray * ( FcollzX_val_MINI[index_right + 1] - FcollzX_val_MINI[index_right]);
                    Splined_Fcollzpp_X_mean_MINI = Splined_Fcollzpp_X_mean_MINI_left + (log10_Mcrit_LW_ave - log10_Mcrit_LW_ave_table_fcollz) / LOG10MTURN_INT * ( Splined_Fcollzpp_X_mean_MINI_right - Splined_Fcollzpp_X_mean_MINI_left );
                    if (Splined_Fcollzpp_X_mean_MINI < 0.) Splined_Fcollzpp_X_mean_MINI = 1e-40;

                    ST_over_PS_MINI[R_ct] = pow(1.+zpp, -X_RAY_SPEC_INDEX_MINI)*fabs(dzpp_for_evolve);
                    ST_over_PS_MINI[R_ct] *= Splined_Fcollzpp_X_mean_MINI;
#endif
                    SFR_timescale_factor[R_ct] = hubble(zpp)*fabs(dtdz(zpp));
                
                }
                else { 
            
                    // Determining values for the evaluating the interpolation table
                    zpp_gridpoint1_int = (int)floor((zpp - determine_zpp_min)/zpp_bin_width);
                    zpp_gridpoint2_int = zpp_gridpoint1_int + 1;
            
                    zpp_gridpoint1 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint1_int;
                    zpp_gridpoint2 = determine_zpp_min + zpp_bin_width*(float)zpp_gridpoint2_int;
            
                    grad1 = ( zpp_gridpoint2 - zpp )/( zpp_gridpoint2 - zpp_gridpoint1 );
                    grad2 = ( zpp - zpp_gridpoint1 )/( zpp_gridpoint2 - zpp_gridpoint1 );
            
                    sigma_Tmin[R_ct] = Sigma_Tmin_grid[zpp_gridpoint1_int] + grad2*( Sigma_Tmin_grid[zpp_gridpoint2_int] - Sigma_Tmin_grid[zpp_gridpoint1_int] );
            
                    // let's now normalize the total collapse fraction so that the mean is the
                    // Sheth-Torman collapse fraction
                    
                    // Evaluating the interpolation table for the collapse fraction and its derivative
                    for(i=0;i<(dens_Ninterp-1);i++) {
                        dens_grad = 1./( density_gridpoints[i+1][R_ct] - density_gridpoints[i][R_ct] );
                
                        if(!SHORTEN_FCOLL) {
                            fcoll_interp1[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                            fcoll_interp2[i][R_ct] = ( ( fcoll_R_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( fcoll_R_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                        }

                        dfcoll_interp1[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i] )*grad2 )*dens_grad;
                        dfcoll_interp2[i][R_ct] = ( ( dfcoll_dz_grid[R_ct][zpp_gridpoint1_int][i+1] )*grad1 + ( dfcoll_dz_grid[R_ct][zpp_gridpoint2_int][i+1] )*grad2 )*dens_grad;
                    }
            
                    // Using the interpolated values to update arrays of relevant quanties for the IGM spin temperature calculation
                    ST_over_PS[R_ct] = dzpp_for_evolve * pow(1.+zpp, -X_RAY_SPEC_INDEX);
                    ST_over_PS[R_ct] *= ( ST_over_PS_arg_grid[zpp_gridpoint1_int] + grad2*( ST_over_PS_arg_grid[zpp_gridpoint2_int] - ST_over_PS_arg_grid[zpp_gridpoint1_int] ) );
                }
#ifdef MINI_HALO
                lower_int_limit = FMAX(nu_tau_one_approx(zp, zpp, x_e_ave, filling_factor_of_HI_zp, ION_EFF_FACTOR, ION_EFF_FACTOR_MINI, log10_Mcrit_LW_ave), NU_X_THRESH);
#else
                lower_int_limit = FMAX(nu_tau_one_approx(zp, zpp, x_e_ave, filling_factor_of_HI_zp), NU_X_THRESH);
#endif
            
                if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp = 0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly
                
                // set up frequency integral table for later interpolation for the cell's x_e value
                for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++){
                    freq_int_heat_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                    freq_int_ion_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
#ifdef MINI_HALO
                    freq_int_heat_tbl_MINI[x_e_ct][R_ct] = integrate_over_nu_MINI(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                    freq_int_ion_tbl_MINI[x_e_ct][R_ct] = integrate_over_nu_MINI(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
#endif
                    
                    if (COMPUTE_Ts) {
                        freq_int_lya_tbl[x_e_ct][R_ct] = integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
#ifdef MINI_HALO
                        freq_int_lya_tbl_MINI[x_e_ct][R_ct] = integrate_over_nu_MINI(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);
#endif
                    }
                }
            
                // and create the sum over Lya transitions from direct Lyn flux
                sum_lyn[R_ct] = 0;
#ifdef MINI_HALO
                sum_lyn_MINI[R_ct] = 0;
                sum_lyLWn[R_ct] = 0;
                sum_lyLWn_MINI[R_ct] = 0;
#endif
                for (n_ct=NSPEC_MAX; n_ct>=2; n_ct--){
                    if (zpp > zmax(zp, n_ct))
                        continue;
                
                    nuprime = nu_n(n_ct)*(1.+zpp)/(1.0+zp);
#ifdef MINI_HALO
                    sum_lyn[R_ct]  += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
                    sum_lyn_MINI[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);
                    if (nuprime < NU_LW_THRESH / NUIONIZATION)
                        nuprime = NU_LW_THRESH / NUIONIZATION;
                    if (nuprime >= nu_n(n_ct + 1))
                        continue;
                    sum_lyLWn[R_ct]  += spectral_emissivity(nuprime, 3, 2);
                    sum_lyLWn_MINI[R_ct] += spectral_emissivity(nuprime, 3, 3);
#else
                    sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0);
#endif
                }
            } // end loop over R_ct filter steps
            
            // Calculate fcoll for each smoothing radius
            
            // Can speed up computation (~20%) by pre-sampling the fcoll field as a function of X_RAY_TVIR_MIN (performed by calling CreateFcollTable.
            // Can be helpful when HII_DIM > ~128, otherwise its easier to just do the full box
            // NOTE: pre-sampling the fcoll field does not support the new parametrization in v1.4.
            
            fcoll_interp_high_min = 1.5;
            fcoll_interp_high_bin_width = 1./((float)NSFR_high-1.)*(Deltac - fcoll_interp_high_min);
            fcoll_interp_high_bin_width_inv = 1./fcoll_interp_high_bin_width;
            
            if(!USE_MASS_DEPENDENT_ZETA) {
                if(SHORTEN_FCOLL) {
                    for(R_ct=0;R_ct<NUM_FILTER_STEPS_FOR_Ts;R_ct++) {
                        fcoll_R = Fcoll_R_Table[counter][Tvir_min_int][R_ct] + ( log10(X_RAY_Tvir_MIN) - ( X_RAY_Tvir_LOWERBOUND + (double)Tvir_min_int*X_RAY_Tvir_BinWidth ) )*( Fcoll_R_Table[counter][Tvir_min_int+1][R_ct] - Fcoll_R_Table[counter][Tvir_min_int][R_ct] )/X_RAY_Tvir_BinWidth;
                    
                        ST_over_PS[R_ct] = ST_over_PS[R_ct]/fcoll_R;
                    }
                }
                else {
                    for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                        for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                            fcoll_R_array[R_ct] += ( fcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct] ) + fcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*( delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct] ) );
                        }
                    }
                    for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                        ST_over_PS[R_ct] = ST_over_PS[R_ct]/(fcoll_R_array[R_ct]/(double)HII_TOT_NUM_PIXELS);
                    }
                }
            }
            // scroll through each cell and update the temperature and residual ionization fraction
            growth_factor_zp = dicke(zp);
            dgrowth_factor_dzp = ddicke_dz(zp);
            dt_dzp = dtdz(zp);
            
            // Conversion of the input bolometric luminosity to a ZETA_X, as used to be used in Ts.c
            // Conversion here means the code otherwise remains the same as the original Ts.c
            if(fabs(X_RAY_SPEC_INDEX - 1.0) < 0.000001) {
                Luminosity_converstion_factor = NU_X_THRESH * log( NU_X_BAND_MAX/NU_X_THRESH );
                Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
            }
            else {
                Luminosity_converstion_factor = pow( NU_X_BAND_MAX , 1. - X_RAY_SPEC_INDEX ) - pow( NU_X_THRESH , 1. - X_RAY_SPEC_INDEX ) ;
                Luminosity_converstion_factor = 1./Luminosity_converstion_factor;
                Luminosity_converstion_factor *= pow( NU_X_THRESH, - X_RAY_SPEC_INDEX )*(1 - X_RAY_SPEC_INDEX);
            }
            // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
            Luminosity_converstion_factor *= (3.1556226e7)/(hplank);
            
            // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
            const_zp_prefactor = ( L_X * Luminosity_converstion_factor ) / NU_X_THRESH * C * F_STAR10 * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1.+zp, X_RAY_SPEC_INDEX+3);
//          This line below is kept purely for reference w.r.t to the original 21cmFAST
//            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1.+zp, X_RAY_SPEC_INDEX+3);
            
#ifdef MINI_HALO
            if(fabs(X_RAY_SPEC_INDEX_MINI - 1.0) < 0.000001) {
                Luminosity_converstion_factor_MINI = NU_X_THRESH * log( NU_X_BAND_MAX/NU_X_THRESH );
                Luminosity_converstion_factor_MINI = 1./Luminosity_converstion_factor_MINI;
            }
            else {
                Luminosity_converstion_factor_MINI = pow( NU_X_BAND_MAX , 1. - X_RAY_SPEC_INDEX_MINI ) - pow( NU_X_THRESH , 1. - X_RAY_SPEC_INDEX_MINI ) ;
                Luminosity_converstion_factor_MINI = 1./Luminosity_converstion_factor_MINI;
                Luminosity_converstion_factor_MINI *= pow( NU_X_THRESH, - X_RAY_SPEC_INDEX_MINI )*(1 - X_RAY_SPEC_INDEX_MINI);
            }
            // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the definition of Luminosity)
            Luminosity_converstion_factor_MINI *= (3.1556226e7)/(hplank);
            
            // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
            const_zp_prefactor_MINI = ( L_X_MINI * Luminosity_converstion_factor_MINI ) / NU_X_THRESH * C * F_STAR10_MINI * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1.+zp, X_RAY_SPEC_INDEX_MINI+3);
//          This line below is kept purely for reference w.r.t to the original 21cmFAST
//            const_zp_prefactor = ZETA_X * X_RAY_SPEC_INDEX / NU_X_THRESH * C * F_STAR * OMb * RHOcrit * pow(CMperMPC, -3) * pow(1.+zp, X_RAY_SPEC_INDEX+3);
#endif
            //////////////////////////////  LOOP THROUGH BOX //////////////////////////////
        
            J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = 0.;
#ifdef MINI_HALO
            J_alpha_ave_MINI = J_LW_ave = J_LW_ave_MINI = Xheat_ave_MINI = 0.;
#endif
        
            // Extra pre-factors etc. are defined here, as they are independent of the density field, and only have to be computed once per z' or R_ct, rather than each box_ct
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                zpp_integrand = ( pow(1.+zp,2)*(1.+zpp_for_evolve_list[R_ct]) )/( pow(1.+zpp_for_evolve_list[R_ct], -X_RAY_SPEC_INDEX) );
                dstarlya_dt_prefactor[R_ct]  = zpp_integrand * sum_lyn[R_ct];
#ifdef MINI_HALO
                zpp_integrand_MINI = ( pow(1.+zp,2)*(1.+zpp_for_evolve_list[R_ct]) )/( pow(1.+zpp_for_evolve_list[R_ct], -X_RAY_SPEC_INDEX_MINI) );
                dstarlya_dt_prefactor_MINI[R_ct]  = zpp_integrand_MINI * sum_lyn_MINI[R_ct];
                dstarlyLW_dt_prefactor[R_ct]  = zpp_integrand * sum_lyLWn[R_ct];
                dstarlyLW_dt_prefactor_MINI[R_ct]  = zpp_integrand_MINI * sum_lyLWn_MINI[R_ct];
#endif
            }
            
            // Required quantities for calculating the IGM spin temperature
            // Note: These used to be determined in evolveInt (and other functions). But I moved them all here, into a single location.
            Trad_fast = T_cmb*(1.0+zp);
            Trad_fast_inv = 1.0/Trad_fast;
            TS_prefactor = pow(1.0e-7*(1.342881e-7 / hubble(zp))*No*pow(1.+zp,3),1./3.);
            xa_tilde_prefactor = 1.66e11/(1.0+zp);
        
            xc_inverse =  pow(1.0+zp,3.0)*T21/( Trad_fast*A10_HYPERFINE );
        
            dcomp_dzp_prefactor = (-1.51e-4)/(hubble(zp)/Ho)/hlittle*pow(Trad_fast,4.0)/(1.0+zp);
        
            prefactor_1 = N_b0 * pow(1.+zp, 3);
            prefactor_2 = F_STAR10 * C * N_b0 / FOURPI;
#ifdef MINI_HALO
            prefactor_2_MINI = F_STAR10_MINI * C * N_b0 / FOURPI;
#endif
        
            x_e_ave = 0; Tk_ave = 0; Ts_ave = 0;

            // Note: I have removed the call to evolveInt, as is default in the original Ts.c. Removal of evolveInt and moving that computation below, removes unneccesary repeated computations
            // and allows for the interpolation tables that are now used to be more easily computed

            // Can precompute these quantities, independent of the density field (i.e. box_ct)
            for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
                for (i=0; i<(x_int_NXHII-1); i++) {
                    m_xHII_low = i;
                    m_xHII_high = m_xHII_low + 1;
            
                    inverse_diff[i] = 1./(x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
                    freq_int_heat_tbl_diff[i][R_ct] = freq_int_heat_tbl[m_xHII_high][R_ct] - freq_int_heat_tbl[m_xHII_low][R_ct];
                    freq_int_ion_tbl_diff[i][R_ct] = freq_int_ion_tbl[m_xHII_high][R_ct] - freq_int_ion_tbl[m_xHII_low][R_ct];
                    freq_int_lya_tbl_diff[i][R_ct] = freq_int_lya_tbl[m_xHII_high][R_ct] - freq_int_lya_tbl[m_xHII_low][R_ct];
#ifdef MINI_HALO
                    freq_int_heat_tbl_diff_MINI[i][R_ct] = freq_int_heat_tbl_MINI[m_xHII_high][R_ct] - freq_int_heat_tbl_MINI[m_xHII_low][R_ct];
                    freq_int_ion_tbl_diff_MINI[i][R_ct] = freq_int_ion_tbl_MINI[m_xHII_high][R_ct] - freq_int_ion_tbl_MINI[m_xHII_low][R_ct];
                    freq_int_lya_tbl_diff_MINI[i][R_ct] = freq_int_lya_tbl_MINI[m_xHII_high][R_ct] - freq_int_lya_tbl_MINI[m_xHII_low][R_ct];
#endif
                    
                }
            }
            
            // Main loop over the entire box for the IGM spin temperature and relevant quantities.
            // The loop ordering is done two different ways for a mass dependent or mass independent zeta
            // The chosen ordering is to minimise both memory footprint and computation time.
            if(USE_MASS_DEPENDENT_ZETA) {
                
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                
                    SFR_for_integrals_Rct[box_ct] = 0.;
                    
                    dxheat_dt_box[box_ct] = 0.;
                    dxion_source_dt_box[box_ct] = 0.;
                    dxlya_dt_box[box_ct] = 0.;
                    dstarlya_dt_box[box_ct] = 0.;
#ifdef MINI_HALO
                    dstarlyLW_dt_box[box_ct] = 0.;

                    SFR_for_integrals_Rct_MINI[box_ct] = 0.;
                    
                    dxheat_dt_box_MINI[box_ct] = 0.;
                    dxion_source_dt_box_MINI[box_ct] = 0.;
                    dxlya_dt_box_MINI[box_ct] = 0.;
                    dstarlya_dt_box_MINI[box_ct] = 0.;
                    dstarlyLW_dt_box_MINI[box_ct] = 0.;
#endif
                    
                    xHII_call = x_e_box[box_ct];
                    
                    // Check if ionized fraction is within boundaries; if not, adjust to be within
                    if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                        xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                    } else if (xHII_call < x_int_XHII[0]) {
                        xHII_call = 1.001*x_int_XHII[0];
                    }
                    //interpolate to correct nu integral value based on the cell's ionization state
                    
                    m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
                    
                    inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]])*inverse_diff[m_xHII_low_box[box_ct]];
                    
                }
                
                for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                    
                    fcoll_interp_min = log10(1. + min_densities[R_ct]*zpp_growth[R_ct]);
                    if( max_densities[R_ct]*zpp_growth[R_ct] > 1.5 ) {
                        fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+1.5)-fcoll_interp_min);
                    }
                    else {
                        fcoll_interp_bin_width = 1./((float)NSFR_low-1.)*(log10(1.+max_densities[R_ct]*zpp_growth[R_ct])-fcoll_interp_min);
                    }
                    fcoll_interp_bin_width_inv = 1./fcoll_interp_bin_width;
                    
                    ave_fcoll = ave_fcoll_inv = 0.0;
#ifdef MINI_HALO
                    ave_fcoll_MINI= ave_fcoll_inv_MINI = 0.0;
#endif
                    
                    for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                        if (!COMPUTE_Ts && (Tk_box[box_ct] > MAX_TK)) //just leave it alone and go to next value
                            continue;
                    
                        curr_dens = delNL0[R_ct][box_ct]*zpp_growth[R_ct];
#ifdef MINI_HALO
                        log10_Mcrit_LW_val = ( log10_Mcrit_LW[R_ct][box_ct] - LOG10MTURN_MIN) / LOG10MTURN_INT;
                        log10_Mcrit_LW_int = (int)floorf( log10_Mcrit_LW_val );
                        log10_Mcrit_LW_diff = log10_Mcrit_LW_val - (float)log10_Mcrit_LW_int;
#endif
                        
                        // Note: Be careful how I have defined this. I have done this for optimisation rather than readability
                        if (!NO_LIGHT){
                            // Now determine all the differentials for the heating/ionisation rate equations
                            
                            if (curr_dens < 1.5){
                                
                                if (curr_dens < -1.) {
                                    fcoll = 0;
#ifdef MINI_HALO
                                    fcoll_MINI = 0;
#endif
                                }
                                else {
                                    dens_val = (log10f(curr_dens+1.) - fcoll_interp_min)*fcoll_interp_bin_width_inv;
                                        
                                    dens_int = (int)floorf( dens_val );
                                    dens_diff = dens_val - (float)dens_int;
                                    
                                    fcoll = log10_Fcollz_SFR_Xray_low_table[counter][R_ct][dens_int]*(1.-dens_diff) + log10_Fcollz_SFR_Xray_low_table[counter][R_ct][dens_int+1]*dens_diff;
                                    
                                    // Note here, this returns the collapse fraction
                                    // The interpolation table is log(10)*exponent, thus exp(log(10)*exponent) = 10^exponent. Which is the value of the collapse fraction.
                                    // The log(10) is implicitly in the interpolation table already
                                    fcoll = expf(fcoll);
#ifdef MINI_HALO
                                    index_left = dens_int+log10_Mcrit_LW_int*NSFR_low;
                                    index_right = dens_int+(log10_Mcrit_LW_int+1)*NSFR_low;
                                    fcoll_MINI_left = log10_Fcollz_SFR_Xray_low_table_MINI[counter][R_ct][index_left]*(1.-dens_diff) + log10_Fcollz_SFR_Xray_low_table_MINI[counter][R_ct][index_left+1]*dens_diff;
                                    fcoll_MINI_right = log10_Fcollz_SFR_Xray_low_table_MINI[counter][R_ct][index_right]*(1.-dens_diff) + log10_Fcollz_SFR_Xray_low_table_MINI[counter][R_ct][index_right+1]*dens_diff;
                                    fcoll_MINI = fcoll_MINI_left * (1.-log10_Mcrit_LW_diff) + fcoll_MINI_right * log10_Mcrit_LW_diff;
                                    fcoll_MINI = expf(fcoll_MINI);
#endif
                                }
                            }
                            else {
                                if (curr_dens < 0.9*Deltac) {
                                        
                                    dens_val = (curr_dens - fcoll_interp_high_min)*fcoll_interp_high_bin_width_inv;
                                    
                                    dens_int = (int)floorf( dens_val );
                                    dens_diff = dens_val - (float)dens_int;

                                    fcoll = Fcollz_SFR_Xray_high_table[counter][R_ct][dens_int]*(1.-dens_diff) + Fcollz_SFR_Xray_high_table[counter][R_ct][dens_int+1]*dens_diff;
#ifdef MINI_HALO
                                    index_left = dens_int+log10_Mcrit_LW_int*NSFR_high;
                                    index_right = dens_int+(log10_Mcrit_LW_int+1)*NSFR_high;
                                    fcoll_MINI_left = Fcollz_SFR_Xray_high_table_MINI[counter][R_ct][index_left]*(1.-dens_diff) + Fcollz_SFR_Xray_high_table_MINI[counter][R_ct][index_left+1]*dens_diff;
                                    fcoll_MINI_right = Fcollz_SFR_Xray_high_table_MINI[counter][R_ct][index_right]*(1.-dens_diff) + Fcollz_SFR_Xray_high_table_MINI[counter][R_ct][index_right+1]*dens_diff;
                                    fcoll_MINI = fcoll_MINI_left * (1 + (float)log10_Mcrit_LW_int - log10_Mcrit_LW_val) + fcoll_MINI_right * (log10_Mcrit_LW_val - (float)log10_Mcrit_LW_int);
#endif

                                }
                                else {
                                    // This is to account for the off-set used in the interpolation table to keep it a float rather than a double.
                                    // The fraction of voxels that are able to achieve this value are extremely low, so not concerned about speed in this instance.
                                    fcoll = pow(10.,10.);
//                                    fcoll = 1.;
#ifdef MINI_HALO
                                    fcoll_MINI = pow(10.,10.);
#endif
                                }
                            }
                            ave_fcoll += fcoll;
                            
                            SFR_for_integrals_Rct[box_ct] = (1.+curr_dens)*fcoll;
#ifdef MINI_HALO
                            ave_fcoll_MINI += fcoll_MINI;
                            
                            SFR_for_integrals_Rct_MINI[box_ct] = (1.+curr_dens)*fcoll_MINI;
#endif
                            
                        }
                    }
                    
                    // Finding the average f_coll for the smoothing radius required dividing through by the 10^10 offset I have used for the tables.
                    ave_fcoll /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);

                    if(ave_fcoll!=0.) {
                        ave_fcoll_inv = 1./ave_fcoll;
                    }


                    // Again, 10^10 accounts for the interpolation table offset.
                    dfcoll_dz_val = (ave_fcoll_inv/pow(10.,10.))*ST_over_PS[R_ct]*SFR_timescale_factor[R_ct]/t_STAR;
                    
                    dstarlya_dt_prefactor[R_ct] *= dfcoll_dz_val;
                    
#ifdef MINI_HALO
                    dstarlyLW_dt_prefactor[R_ct] *= dfcoll_dz_val;

                    ave_fcoll_MINI /= (pow(10.,10.)*(double)HII_TOT_NUM_PIXELS);

                    if(ave_fcoll_MINI!=0.) {
                        ave_fcoll_inv_MINI = 1./ave_fcoll_MINI;
                    }

                    dfcoll_dz_val_MINI = (ave_fcoll_inv_MINI/pow(10.,10.))*ST_over_PS_MINI[R_ct]*SFR_timescale_factor[R_ct]/t_STAR;
                    
                    dstarlya_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
                    dstarlyLW_dt_prefactor_MINI[R_ct] *= dfcoll_dz_val_MINI;
#endif
                    for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                        
                        if (!COMPUTE_Ts && (Tk_box[box_ct] > MAX_TK)) //just leave it alone and go to next value
                            continue;
                        
                        // I've added the addition of zero just in case. It should be zero anyway, but just in case there is some weird
                        // numerical thing
                        if(ave_fcoll!=0.) {
                            dxheat_dt_box[box_ct] += (dfcoll_dz_val*(double)SFR_for_integrals_Rct[box_ct]*( (freq_int_heat_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_heat_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            dxion_source_dt_box[box_ct] += (dfcoll_dz_val*(double)SFR_for_integrals_Rct[box_ct]*( (freq_int_ion_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_ion_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                            
                        }
                        else {
                            dxheat_dt_box[box_ct] += 0.;
                            dxion_source_dt_box[box_ct] += 0.;
                        }
                        
                        if (COMPUTE_Ts){
                            if(ave_fcoll!=0.) {
                                dxlya_dt_box[box_ct] += (dfcoll_dz_val*(double)SFR_for_integrals_Rct[box_ct]*( (freq_int_lya_tbl_diff[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_lya_tbl[m_xHII_low_box[box_ct]][R_ct] ));
                                dstarlya_dt_box[box_ct] += (double)SFR_for_integrals_Rct[box_ct]*dstarlya_dt_prefactor[R_ct];
#ifdef MINI_HALO
                                dstarlyLW_dt_box[box_ct] += (double)SFR_for_integrals_Rct[box_ct]*dstarlyLW_dt_prefactor[R_ct];;
#endif
                                
                            }
                            else {
                                dxlya_dt_box[box_ct] += 0.;
                                dstarlya_dt_box[box_ct] += 0.;
#ifdef MINI_HALO
                                dstarlyLW_dt_box[box_ct] += 0.;
#endif
                            }
                        }
#ifdef MINI_HALO
                        if(ave_fcoll_MINI!=0.) {
                            dxheat_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)SFR_for_integrals_Rct_MINI[box_ct]*( (freq_int_heat_tbl_diff_MINI[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_heat_tbl_MINI[m_xHII_low_box[box_ct]][R_ct] ));
                            dxion_source_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)SFR_for_integrals_Rct_MINI[box_ct]*( (freq_int_ion_tbl_diff_MINI[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_ion_tbl_MINI[m_xHII_low_box[box_ct]][R_ct] ));
                            
                        }
                        else {
                            dxheat_dt_box_MINI[box_ct] += 0.;
                            dxion_source_dt_box_MINI[box_ct] += 0.;
                        }
                        
                        if (COMPUTE_Ts){
                            if(ave_fcoll_MINI!=0.) {
                                dxlya_dt_box_MINI[box_ct] += (dfcoll_dz_val_MINI*(double)SFR_for_integrals_Rct_MINI[box_ct]*( (freq_int_lya_tbl_diff_MINI[m_xHII_low_box[box_ct]][R_ct])*inverse_val_box[box_ct] + freq_int_lya_tbl_MINI[m_xHII_low_box[box_ct]][R_ct] ));
                                dstarlya_dt_box_MINI[box_ct] += (double)SFR_for_integrals_Rct_MINI[box_ct]*dstarlya_dt_prefactor_MINI[R_ct];
                                dstarlyLW_dt_box_MINI[box_ct] += (double)SFR_for_integrals_Rct_MINI[box_ct]*dstarlyLW_dt_prefactor_MINI[R_ct];;
                                
                            }
                            else {
                                dxlya_dt_box_MINI[box_ct] += 0.;
                                dstarlya_dt_box_MINI[box_ct] += 0.;
                                dstarlyLW_dt_box_MINI[box_ct] += 0.;
                            }
                        }
#endif
                        
                        // If R_ct == 0, as this is the final smoothing scale (i.e. it is reversed)
                        if(R_ct==0) {
                            
                            x_e = x_e_box[box_ct];
                            T = Tk_box[box_ct];
                            
                            // add prefactors
                            dxheat_dt_box[box_ct] *= const_zp_prefactor;
                            dxion_source_dt_box[box_ct] *= const_zp_prefactor;
                            if (COMPUTE_Ts){
                                dxlya_dt_box[box_ct] *= const_zp_prefactor*prefactor_1 * (1.+delNL0[0][box_ct]*growth_factor_zp);
                                dstarlya_dt_box[box_ct] *= prefactor_2;
#ifdef MINI_HALO
                                dstarlyLW_dt_box[box_ct] *= prefactor_2 * (hplank * 1e21);
#endif
                            }
#ifdef MINI_HALO
                            dxheat_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;
                            dxion_source_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI;
                            if (COMPUTE_Ts){
                                dxlya_dt_box_MINI[box_ct] *= const_zp_prefactor_MINI*prefactor_1 * (1.+delNL0[0][box_ct]*growth_factor_zp);
                                dstarlya_dt_box_MINI[box_ct] *= prefactor_2_MINI;
                                dstarlyLW_dt_box_MINI[box_ct] *= prefactor_2_MINI * (hplank * 1e21);
                            }
#endif
                            
                            // Now we can solve the evolution equations  //
                            
                            // First let's do dxe_dzp //
                            dxion_sink_dt = alpha_A(T) * CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * (1.+delNL0[0][box_ct]*growth_factor_zp);
#ifdef MINI_HALO
                            dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] + dxion_source_dt_box_MINI[box_ct] - dxion_sink_dt );
#else
                            dxe_dzp = dt_dzp*(dxion_source_dt_box[box_ct] - dxion_sink_dt );
#endif
                            
                            // Next, let's get the temperature components //
                            // first, adiabatic term
                            dadia_dzp = 3/(1.0+zp);
                            if (fabs(delNL0[0][box_ct]) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                dadia_dzp += dgrowth_factor_dzp/(1.0/delNL0[0][box_ct]+growth_factor_zp);
                            
                            dadia_dzp *= (2.0/3.0)*T;
                            
                            // next heating due to the changing species
                            dspec_dzp = - dxe_dzp * T / (1.+x_e);
                            
                            // next, Compton heating
                            //                dcomp_dzp = dT_comp(zp, T, x_e);
                            dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e+f_He))*( Trad_fast - T );
                            
                            // lastly, X-ray heating
                            dxheat_dzp = dxheat_dt_box[box_ct] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
#ifdef MINI_HALO
                            dxheat_dzp_MINI = dxheat_dt_box_MINI[box_ct]* dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
#endif
                            
                            //update quantities
                            x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                            if (x_e > 1) // can do this late in evolution if dzp is too large
                                x_e = 1 - FRACT_FLOAT_ERR;
                            else if (x_e < 0)
                                x_e = 0;
                            if (T < MAX_TK) {
                                T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp ) * dzp;
#ifdef MINI_HALO
                                T += dxheat_dzp_MINI * dzp;
#endif
                            }
                            
                            if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                T = T_cmb*(1.+zp);
                            }

                            x_e_box[box_ct] = x_e;
                            Tk_box[box_ct] = T;

                            if (COMPUTE_Ts){
                                J_alpha_tot = dxlya_dt_box[box_ct] + dstarlya_dt_box[box_ct]; //not really d/dz, but the lya flux
#ifdef MINI_HALO
                                J_alpha_tot_MINI = dxlya_dt_box_MINI[box_ct] + dstarlya_dt_box_MINI[box_ct]; //not really d/dz, but the lya flux
                                J_21_LW[box_ct] = dstarlyLW_dt_box[box_ct] + dstarlyLW_dt_box_MINI[box_ct];
#endif

                                // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                                // Algorithm is the same, but written to be more computationally efficient
                                T_inv = expf((-1.)*logf(T));
                                T_inv_sq = expf((-2.)*logf(T));
                                
                                xc_fast = (1.0+delNL0[0][box_ct]*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10_float(T,0) + x_e*N_b0*kappa_10_elec_float(T,0) + x_e*No*kappa_10_pH_float(T,0) );

                                xi_power = TS_prefactor * cbrt((1.0+delNL0[0][box_ct]*growth_factor_zp)*(1.0-x_e)*T_inv_sq);
                                xa_tilde_fast_arg = xa_tilde_prefactor*(J_alpha_tot+J_alpha_tot_MINI)*pow( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power, -1. );
                            
                                // New in v1.4
                                if (fabs(J_alpha_tot+J_alpha_tot_MINI) > 1.0e-20) { // Must use WF effect
                                    TS_fast = Trad_fast;
                                    TSold_fast = 0.0;
                                    while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {
                                        
                                        TSold_fast = TS_fast;
                                        
                                        xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - 0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                                        
                                        TS_fast = (1.0+xa_tilde_fast+xc_fast)*pow(Trad_fast_inv+xa_tilde_fast*( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                                    }
                                } else { // Collisions only
                                    TS_fast = (1.0 + xc_fast)/(Trad_fast_inv + xc_fast*T_inv);
                                    xa_tilde_fast = 0.0;
                                }
                                if(TS_fast < 0.) {
                                    // It can very rarely result in a negative spin temperature. If negative, it is a very small number. Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                                    TS_fast = fabs(TS_fast);
                                }

                                Ts[box_ct] = TS_fast;
                                
                                if(STORE_DATA || OUTPUT_AVE) {
                                    J_alpha_ave += J_alpha_tot;
                                    xalpha_ave += xa_tilde_fast;
                                    Xheat_ave += ( dxheat_dzp );
                                    Xion_ave += ( dt_dzp*dxion_source_dt_box[box_ct] );
                                    Ts_ave += TS_fast;
                                    Tk_ave += T;
#ifdef MINI_HALO
                                    J_alpha_ave_MINI += J_alpha_tot_MINI;
                                    Xheat_ave_MINI += ( dxheat_dzp_MINI );
                                    J_LW_ave += dstarlyLW_dt_box[box_ct];
                                    J_LW_ave_MINI += dstarlyLW_dt_box_MINI[box_ct];
#endif
                                }
                            }
                            x_e_ave += x_e;
                        }
                    }
                }
                
            }
            else {
                
                
                // Main loop over the entire box for the IGM spin temperature and relevant quantities.
                for (box_ct=HII_TOT_NUM_PIXELS; box_ct--;){
                    if (!COMPUTE_Ts && (Tk_box[box_ct] > MAX_TK)) //just leave it alone and go to next value
                        continue;
                    
                    x_e = x_e_box[box_ct];
                    T = Tk_box[box_ct];
                    
                    xHII_call = x_e;
                    
                    // Check if ionized fraction is within boundaries; if not, adjust to be within
                    if (xHII_call > x_int_XHII[x_int_NXHII-1]*0.999) {
                        xHII_call = x_int_XHII[x_int_NXHII-1]*0.999;
                    } else if (xHII_call < x_int_XHII[0]) {
                        xHII_call = 1.001*x_int_XHII[0];
                    }
                    //interpolate to correct nu integral value based on the cell's ionization state
                    
                    m_xHII_low = locate_xHII_index(xHII_call);
                    
                    inverse_val = (xHII_call - x_int_XHII[m_xHII_low])*inverse_diff[m_xHII_low];
                    
                    // First, let's do the trapazoidal integration over zpp
                    dxheat_dt = 0;
                    dxion_source_dt = 0;
                    dxlya_dt = 0;
                    dstarlya_dt = 0;
                    
                    curr_delNL0 = delNL0_rev[box_ct][0];
                    
                    if (!NO_LIGHT){
                        // Now determine all the differentials for the heating/ionisation rate equations
                        for (R_ct=NUM_FILTER_STEPS_FOR_Ts; R_ct--;){
                            
                            dfcoll_dz_val = ST_over_PS[R_ct]*(1.+delNL0_rev[box_ct][R_ct]*zpp_growth[R_ct])*( dfcoll_interp1[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(density_gridpoints[dens_grid_int_vals[box_ct][R_ct] + 1][R_ct] - delNL0_rev[box_ct][R_ct]) + dfcoll_interp2[dens_grid_int_vals[box_ct][R_ct]][R_ct]*(delNL0_rev[box_ct][R_ct] - density_gridpoints[dens_grid_int_vals[box_ct][R_ct]][R_ct]) );
                        }
                            
                        dxheat_dt += dfcoll_dz_val * ( (freq_int_heat_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_heat_tbl[m_xHII_low][R_ct] );
                        dxion_source_dt += dfcoll_dz_val * ( (freq_int_ion_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_ion_tbl[m_xHII_low][R_ct] );
                            
                        if (COMPUTE_Ts){
                            dxlya_dt += dfcoll_dz_val * ( (freq_int_lya_tbl_diff[m_xHII_low][R_ct])*inverse_val + freq_int_lya_tbl[m_xHII_low][R_ct] );
                            dstarlya_dt += dfcoll_dz_val*dstarlya_dt_prefactor[R_ct];
                        }
                    }
                    
                    // add prefactors
                    dxheat_dt *= const_zp_prefactor;
                    dxion_source_dt *= const_zp_prefactor;
                    if (COMPUTE_Ts){
                        dxlya_dt *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                        dstarlya_dt *= prefactor_2;
                    }
                    
                    // Now we can solve the evolution equations  //
                    
                    // First let's do dxe_dzp //
                    dxion_sink_dt = alpha_A(T) * CLUMPING_FACTOR * x_e*x_e * f_H * prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                    dxe_dzp = dt_dzp*(dxion_source_dt - dxion_sink_dt );
                    
                    // Next, let's get the temperature components //
                    // first, adiabatic term
                    dadia_dzp = 3/(1.0+zp);
                    if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                        dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);
                    
                    dadia_dzp *= (2.0/3.0)*T;
                    
                    // next heating due to the changing species
                    dspec_dzp = - dxe_dzp * T / (1.+x_e);
                    
                    // next, Compton heating
                    //                dcomp_dzp = dT_comp(zp, T, x_e);
                    dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e+f_He))*( Trad_fast - T );
                    
                    // lastly, X-ray heating
                    dxheat_dzp = dxheat_dt * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                    
                    //update quantities
                    x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                    if (x_e > 1) // can do this late in evolution if dzp is too large
                        x_e = 1 - FRACT_FLOAT_ERR;
                    else if (x_e < 0)
                        x_e = 0;
                    if (T < MAX_TK) {
                        T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp ) * dzp;
                    }
                    
                    if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                        T = T_cmb*(1.+zp);
                    }
                    
                    x_e_box[box_ct] = x_e;
                    Tk_box[box_ct] = T;
                    
                    if (COMPUTE_Ts){
                        J_alpha_tot = ( dxlya_dt + dstarlya_dt ); //not really d/dz, but the lya flux
                        
                        // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                        // Algorithm is the same, but written to be more computationally efficient
                        T_inv = pow(T,-1.);
                        T_inv_sq = pow(T,-2.);
                        
                        xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*( (1.0-x_e)*No*kappa_10_float(T,0) + x_e*N_b0*kappa_10_elec_float(T,0) + x_e*No*kappa_10_pH_float(T,0) );
                        xi_power = TS_prefactor * pow((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq, 1.0/3.0);
                        xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot*pow( 1.0 + 2.98394*xi_power + 1.53583*pow(xi_power,2.) + 3.85289*pow(xi_power,3.), -1. );
                        
                        // New in v1.4
                        if (fabs(J_alpha_tot) > 1.0e-20) { // Must use WF effect
                            TS_fast = Trad_fast;
                            TSold_fast = 0.0;
                            while (fabs(TS_fast-TSold_fast)/TS_fast > 1.0e-3) {
                                
                                TSold_fast = TS_fast;
                                
                                xa_tilde_fast = ( 1.0 - 0.0631789*T_inv + 0.115995*T_inv_sq - 0.401403*T_inv*pow(TS_fast,-1.) + 0.336463*T_inv_sq*pow(TS_fast,-1.) )*xa_tilde_fast_arg;
                                
                                TS_fast = (1.0+xa_tilde_fast+xc_fast)*pow(Trad_fast_inv+xa_tilde_fast*( T_inv + 0.405535*T_inv*pow(TS_fast,-1.) - 0.405535*T_inv_sq ) + xc_fast*T_inv,-1.);
                            }
                        }
                        else { // Collisions only
                            TS_fast = (1.0 + xc_fast)/(Trad_fast_inv + xc_fast*T_inv);
                            xa_tilde_fast = 0.0;
                        }
                        if(TS_fast < 0.) {
                            // It can very rarely result in a negative spin temperature. If negative, it is a very small number. Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                            TS_fast = fabs(TS_fast);
                        }
                        
                        Ts[box_ct] = TS_fast;
                        
                        if(OUTPUT_AVE) {
                            J_alpha_ave += J_alpha_tot;
                            xalpha_ave += xa_tilde_fast;
                            Xheat_ave += ( dxheat_dzp );
                            Xion_ave += ( dt_dzp*dxion_source_dt );
                            Ts_ave += TS_fast;
                            Tk_ave += T;
                        }
                    }
                    x_e_ave += x_e;
                }
                
                
            }
            
            // For this redshift snapshot, we now determine the ionisation field and subsequently the 21cm brightness temperature map (also the 21cm PS)
            // Note the relatively small tolerance for zp and the input redshift. The user needs to be careful to provide the correct redshifts for evaluating this to high precision.
            // If the light-cone option is set, this criterion should automatically be met
            for(i_z=0;i_z<N_USER_REDSHIFT;i_z++) {
                if(fabs(redshifts[i_z] - zp)<0.001) {
                    
                    memcpy(Ts_z,Ts,sizeof(float)*HII_TOT_NUM_PIXELS);
                    memcpy(x_e_z,x_e_box,sizeof(float)*HII_TOT_NUM_PIXELS);
                    
                    if(i_z==0) {
                        // If in here, it doesn't matter what PREV_REDSHIFT is set to
                        // as the recombinations will not be calculated
                        nf_ave = ComputeIonisationBoxes(i_z,redshifts[i_z],redshifts[i_z]+0.2);
                    }
                    else {
                        nf_ave = ComputeIonisationBoxes(i_z,redshifts[i_z],redshifts[i_z-1]);
                    }
                    break;
                }
            }
            
            /////////////////////////////  END LOOP ////////////////////////////////////////////
        
            // compute new average values
            x_e_ave /= (double)HII_TOT_NUM_PIXELS;
            
            if(STORE_DATA || OUTPUT_AVE) {
                Ts_ave /= (double)HII_TOT_NUM_PIXELS;
                Tk_ave /= (double)HII_TOT_NUM_PIXELS;
                J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
                xalpha_ave /= (double)HII_TOT_NUM_PIXELS;
                Xheat_ave /= (double)HII_TOT_NUM_PIXELS;
                Xion_ave /= (double)HII_TOT_NUM_PIXELS;
#ifdef MINI_HALO
                J_alpha_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                Xheat_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
                J_LW_ave_MINI /= (double)HII_TOT_NUM_PIXELS;
                aveJ_21_LW[i_z] = J_LW_ave;
                aveJ_21_LW_MINI[i_z] = J_LW_ave_MINI;
                aveJ_alpha[i_z] = J_alpha_ave;
                aveJ_alpha_MINI[i_z] = J_alpha_ave_MINI;
                aveXheat[i_z] = Xheat_ave;
                aveXheat_MINI[i_z] = Xheat_ave_MINI;
                aveNion[i_z] = Splined_Fcollzp_mean * ION_EFF_FACTOR;
                aveNion_MINI[i_z] = Splined_Fcollzp_mean_MINI * ION_EFF_FACTOR_MINI;
#endif
            }
        
#ifdef MINI_HALO
            if(OUTPUT_AVE) {
                printf("zp = %.2f nf_ave = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e (%e) xalpha_ave = %e Xheat_ave = %e (%e) Xion_ave = %e J_LW_ave = %e (%e)\n",zp,nf_ave,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,J_alpha_ave_MINI,xalpha_ave,Xheat_ave,Xheat_ave_MINI,Xion_ave,J_LW_ave*1e-21, J_LW_ave_MINI * 1e-21);
#else
                printf("zp = %.2f nf_ave = %e Ts_ave = %e x_e_ave = %e Tk_ave = %e J_alpha_ave = %e xalpha_ave = %e Xheat_ave = %e Xion_ave = %e\n",zp,nf_ave,Ts_ave,x_e_ave,Tk_ave,J_alpha_ave,xalpha_ave,Xheat_ave,Xion_ave);
#endif
            }
            
            prev_zp = zp;
            zp = ((1.+prev_zp) / ZPRIME_STEP_FACTOR - 1);
            dzp = zp - prev_zp;
            
            counter += 1;
            
        } // end main integral loop over z'
        
        destroy_21cmMC_Ts_arrays();
        destruct_heat();
#ifdef MINI_HALO
        for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
            free(log10_Mcrit_LW[R_ct]); 
        }
#endif
    }
    
    if(!USE_MASS_DEPENDENT_ZETA) {
        for(i=0;i<Numzp_for_table;i++) {
            for(j=0;j<X_RAY_Tvir_POINTS;j++) {
                free(Fcoll_R_Table[i][j]);
            }
            free(Fcoll_R_Table[i]);
        }
        free(Fcoll_R_Table);
    }
}

float ComputeIonisationBoxes(int sample_index, float REDSHIFT_SAMPLE, float PREV_REDSHIFT) {
    
    /* This is an entire re-write of find_HII_bubbles.c from 21cmFAST. Refer back to that code if this becomes a little confusing, the computation and algorithm are the same.
     Additionally, the code here includes delta_T.c for calculating the 21cm PS, and also redshift_interpolate_boxes.c for calculating the lightcones. */
    
    char filename[500];
    char wisdom_filename[500];
    FILE *F;
    fftwf_plan plan;
    
    // Other parameters used in the code
    int i,j,k,ii, x,y,z, N_min_cell, LAST_FILTER_STEP, short_completely_ionised,skip_deallocate,first_step_R;
    int n_x, n_y, n_z,counter,LOOP_INDEX;
    unsigned long long ct;
    
    float growth_factor,MFEEDBACK, R, pixel_mass, cell_length_factor, ave_N_min_cell, M_MIN, nf;
    float f_coll_crit, erfc_denom, erfc_denom_cell, res_xH, Splined_Fcoll, Splined_Fcoll_temp, sqrtarg, xHI_from_xrays, curr_dens, stored_R, massofscaleR, ans;
     
    double global_xH, global_step_xH, ST_over_PS_HII, mean_f_coll_st, f_coll_min, f_coll, f_coll_temp, f_coll_from_table, f_coll_from_table_1, f_coll_from_table_2;
#ifdef MINI_HALO
    float Splined_Fcoll_left, Splined_Fcoll_right;
    float Splined_Fcoll_MINI,Splined_Fcoll_MINI_left, Splined_Fcoll_MINI_right;
    float prev_dens;
    float prev_Splined_Fcoll,prev_Splined_Fcoll_left,prev_Splined_Fcoll_right;
    float prev_Splined_Fcoll_MINI,prev_Splined_Fcoll_MINI_left, prev_Splined_Fcoll_MINI_right;
    double ST_over_PS_HII_MINI, mean_f_coll_st_MINI, f_coll_MINI, f_coll_min_MINI;
    int index_left, index_right;
#endif
    
    double t_ast, dfcolldt, Gamma_R_prefactor, rec, dNrec;
    float growth_factor_dz, fabs_dtdz, ZSTEP, Gamma_R, z_eff;
    const float dz = 0.01;
    
    float redshift_table_fcollz_diff;
    
    int redshift_int_fcollz;
#ifdef MINI_HALO
    double Gamma_R_prefactor_MINI;
    double log10_Mmin, log10_Mmin_MINI;
    float log10_Mmin_ave_table_fcollz, log10_Mmin_MINI_ave_table_fcollz,log10_Mmin_ave_table_fcollz_diff,log10_Mmin_MINI_ave_table_fcollz_diff;
    int log10_Mmin_ave_int_fcollz, log10_Mmin_MINI_ave_int_fcollz;
    double Mcrit_atom, Mcrit_RE, Mcrit_LW, log10_Mcrit_atom, log10_Mcrit_mol, Mmin, Mmin_MINI;
#endif
    
    float ln_10;
    
    ln_10 = log(10);
    
    float dens_val, overdense_small_min, overdense_small_bin_width, overdense_small_bin_width_inv, overdense_large_min, overdense_large_bin_width, overdense_large_bin_width_inv;
    
    int overdense_int;
#ifdef MINI_HALO
    float log10_Mmin_val, log10_Mmin_MINI_val, log10_Mmin_diff, log10_Mmin_MINI_diff, prev_dens_val, prev_dens_diff, dens_diff;
    int   log10_Mmin_int, log10_Mmin_MINI_int, prev_overdense_int;
#endif
    
    overdense_large_min = 1.5*0.999;
    overdense_large_bin_width = 1./((double)NSFR_high-1.)*(Deltac-overdense_large_min);
    overdense_large_bin_width_inv = 1./overdense_large_bin_width;
    
    const gsl_rng_type * T;
    gsl_rng * r;
    
    skip_deallocate = 0;
    
    // Choice of DIM is arbitrary, just needs to be a value larger than HII_DIM. DIM should be sufficient as it shouldn't exceeded DIM (provided DIM > HII_DIM by a factor of at least ~3)
    int *LOS_index = (int*) calloc(DIM,sizeof(int));
    int *slice_index = (int*) calloc(DIM,sizeof(int));
    
    int total_in_z = 0;
    
    float d1_low, d1_high, d2_low, d2_high, gradient_component, min_gradient_component, subcell_width, x_val1, x_val2, subcell_displacement;
    float RSD_pos_new, RSD_pos_new_boundary_low,RSD_pos_new_boundary_high, fraction_within, fraction_outside, cell_distance;
    float Mlim_Fstar, Mlim_Fesc; // New in v1.4
#ifdef MINI_HALO
    float Mlim_Fstar_MINI;
#endif

    float min_density, max_density;
    
    int min_slice_index,slice_index_reducedLC;
    
    min_slice_index = HII_DIM + 1;
    slice_index_reducedLC = 0;
    
    // For recombinations
    if(INHOMO_RECO) {
        ZSTEP = PREV_REDSHIFT - REDSHIFT_SAMPLE;
    }
    else {
        ZSTEP = 0.2;
    }
    fabs_dtdz = fabs(dtdz(REDSHIFT_SAMPLE));
    t_ast = t_STAR * t_hubble(REDSHIFT_SAMPLE);
    growth_factor_dz = dicke(REDSHIFT_SAMPLE-dz);
    
    // if USE_FCOLL_IONISATION_TABLE == 1, we are only calculating the ionisation fraction within a smaller volume (however much is required to linearly interpolate the fields for the light-cone).
    // To know which slices to keep and discard, need to store the relevant indices etc. for the given redshift. The below code does this.
    if(USE_FCOLL_IONISATION_TABLE) {
        
        total_in_z = 0;
        
        // LOS_direction is used in two separate locations (iterated), so we store the state for each individucal usage
        LOS_direction = Stored_LOS_direction_state_2;
        
        // First store the requisite slices for this redshift snapshot.
        if(start_index_LC[N_USER_REDSHIFT-sample_index-1] > end_index_LC[N_USER_REDSHIFT-sample_index-1]) {
            for(ii=0;ii<end_index_LC[N_USER_REDSHIFT-sample_index-1];ii++) {
                LOS_index[total_in_z] = LOS_direction;
                slice_index[total_in_z] = ii;
                total_in_z += 1;
            }
            for(ii=start_index_LC[N_USER_REDSHIFT-sample_index-1];ii<HII_DIM;ii++) {
                LOS_index[total_in_z] = LOS_direction;
                slice_index[total_in_z] = ii;
                total_in_z += 1;
            }
        }
        else {
            for(ii=start_index_LC[N_USER_REDSHIFT-sample_index-1];ii<end_index_LC[N_USER_REDSHIFT-sample_index-1];ii++) {
                LOS_index[total_in_z] = LOS_direction;
                slice_index[total_in_z] = ii;
                total_in_z += 1;
            }
        }
        
        if(sample_index<(N_USER_REDSHIFT-1)) {
            
            // Now store the requisite slices for the next redshift (lower redshift) snapshot
            if(start_index_LC[N_USER_REDSHIFT-(sample_index+1)-1] > end_index_LC[N_USER_REDSHIFT-(sample_index+1)-1]) {
                for(ii=0;ii<end_index_LC[N_USER_REDSHIFT-(sample_index+1)-1];ii++) {
                    LOS_index[total_in_z] = LOS_direction;
                    slice_index[total_in_z] = ii;
                    total_in_z += 1;
                }
                
                for(ii=start_index_LC[N_USER_REDSHIFT-(sample_index+1)-1];ii<HII_DIM;ii++) {
                    LOS_index[total_in_z] = LOS_direction;
                    slice_index[total_in_z] = ii;
                    total_in_z += 1;
                }
            }
            else {
                for(ii=start_index_LC[N_USER_REDSHIFT-(sample_index+1)-1];ii<end_index_LC[N_USER_REDSHIFT-(sample_index+1)-1];ii++) {
                    LOS_index[total_in_z] = LOS_direction;
                    slice_index[total_in_z] = ii;
                    total_in_z += 1;
                }
            }
        }
        if(total_in_z > HII_DIM) {
            // The entire box is used (i.e. we haven't reduced the computation), since the number of required slices between two subsequent redshifts exceeds HII_DIM (i.e. boxes too small).
            total_in_z = HII_DIM;
            for(ii=0;ii<HII_DIM;ii++) {
                slice_index[ii] = ii;
                LOS_index[ii] = Default_LOS_direction;
            }
        }
        
        if((N_USER_REDSHIFT-(sample_index)-1)>0) {
            min_slice_index = start_index_LC[N_USER_REDSHIFT-(sample_index+1)-1];
        }
        else {
            min_slice_index = start_index_LC[N_USER_REDSHIFT-sample_index-1];
        }
        
        if(SUBCELL_RSD) {
            // Add in the padding to the calculation of the ionisation field to account for the fact that cells can enter/exit the reduced LC box.
            min_slice_index = min_slice_index - LC_BOX_PADDING;
            if(min_slice_index < 0) {
                min_slice_index = min_slice_index + HII_DIM;
            }
            total_in_z += 2*LC_BOX_PADDING;
            
            if(total_in_z > HII_DIM) {
                // if after adding in the padding, the total is larger than the box length, then only need to sample at the box length (HII_DIM) and not greater
                total_in_z = HII_DIM;
            }
        }
    }
    
    /////////////////////////////////   BEGIN INITIALIZATION   //////////////////////////////////
    
    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((BOX_LEN > DIM) && !EVOLVE_DENSITY_LINEARLY){
    printf("perturb_field.c: WARNING: Resolution is likely too low for accurate evolved density fields\n It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }
     
    // initialize power spectrum
    growth_factor = dicke(REDSHIFT_SAMPLE);
     
    init_21cmMC_HII_arrays();
    if(GenerateNewICs) {
        
        // Calculate the density field for this redshift if the initial conditions/cosmology are changing
        ComputePerturbField(REDSHIFT_SAMPLE);
            
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)deltax_unfiltered + HII_R_FFT_INDEX(i,j,k)) = LOWRES_density_REDSHIFT[HII_R_INDEX(i,j,k)];
                }
            }
        }
        
    }
    else {
        // Read the desnity field of this redshift from file
        sprintf(filename, "../Boxes/updated_smoothed_deltax_z%06.2f_%i_%.0fMpc", REDSHIFT_SAMPLE, HII_DIM, BOX_LEN);
        F = fopen(filename, "rb");
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    if (fread((float *)deltax_unfiltered + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F)!=1){
                        printf("Read error occured while reading deltax box.\n");
                    }
                }
            }
        }
        fclose(F);
    }

    
    // keep the unfiltered density field in an array, to save it for later
    memcpy(deltax_unfiltered_original, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
     
    i=0;
     
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    
    pixel_mass = RtoM(L_FACTOR*BOX_LEN/(float)HII_DIM);
//    f_coll_crit = 1/HII_EFF_FACTOR;
    cell_length_factor = L_FACTOR;
    
    //set the minimum source mass
    if (USE_MASS_DEPENDENT_ZETA) {
#ifdef MINI_HALO
        Mcrit_atom                 = atomic_cooling_threshold(REDSHIFT_SAMPLE);
        log10_Mcrit_atom           = log10(Mcrit_atom);
        log10_Mcrit_mol            = log10(lyman_werner_threshold(REDSHIFT_SAMPLE, 0.));
        log10_Mmin_ave[sample_index] = 0.;
        log10_Mmin_MINI_ave[sample_index] = 0.;
        for (x=0; x<HII_DIM; x++){
            for (y=0; y<HII_DIM; y++){
                for (z=0; z<HII_DIM; z++){
                    Mcrit_RE        = reionization_feedback(REDSHIFT_SAMPLE, Gamma12[HII_R_INDEX(x, y, z)], z_re[HII_R_INDEX(x, y, z)]);
                    Mcrit_LW        = lyman_werner_threshold(REDSHIFT_SAMPLE, prev_J_21_LW[HII_R_INDEX(x, y, z)]);
                    Mmin            = Mcrit_RE > Mcrit_atom ? Mcrit_RE : Mcrit_atom;
                    Mmin_MINI       = Mcrit_RE > Mcrit_LW   ? Mcrit_RE : Mcrit_LW;
                    log10_Mmin      = log10(Mmin);
                    log10_Mmin_MINI = log10(Mmin_MINI);
                    prev_J_21_LW[HII_R_INDEX(x, y, z)] = J_21_LW[HII_R_INDEX(x, y, z)];

                    *((float *)log10_Mmin_unfiltered + HII_R_FFT_INDEX(x,y,z))      = log10_Mmin;
                    *((float *)log10_Mmin_MINI_unfiltered + HII_R_FFT_INDEX(x,y,z)) = log10_Mmin_MINI;

                    log10_Mmin_ave[sample_index]      += log10_Mmin;
                    log10_Mmin_MINI_ave[sample_index] += log10_Mmin_MINI;
                }
            }
        }
        log10_Mmin_ave[sample_index]      /= HII_TOT_NUM_PIXELS;
        log10_Mmin_MINI_ave[sample_index] /= HII_TOT_NUM_PIXELS;
        Mmin                 = pow(10., log10_Mmin_ave[sample_index]);
        Mmin_MINI            = pow(10., log10_Mmin_MINI_ave[sample_index]);
        M_MIN = 1e5;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16,  ALPHA_STAR, F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, 1e16, ALPHA_ESC, F_ESC10);
        Mlim_Fstar_MINI = Mass_limit_bisection(M_MIN, 1e16,  ALPHA_STAR, F_STAR10_MINI);
#else
        M_MIN = M_TURN/50.;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16,  ALPHA_STAR, F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, 1e16, ALPHA_ESC, F_ESC10);
#endif
    }    
    else {
        M_MIN = M_TURNOVER;
    }
    // check for WDM

    if (P_CUTOFF && ( M_MIN < M_J_WDM())){
        printf( "The default Jeans mass of %e Msun is smaller than the scale supressed by the effective pressure of WDM.\n", M_MIN);
        M_MIN = M_J_WDM();
        printf( "Setting a new effective Jeans mass from WDM pressure supression of %e Msun\n", M_MIN);
    }
     
    for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
        xH[ct] = 1.;
    }
     
    // lets check if we are going to bother with computing the inhmogeneous field at all...
    
    global_xH = 0.0;
    
    // New in v1.4
    if (USE_MASS_DEPENDENT_ZETA) {
        if (USE_LIGHTCONE || USE_TS_FLUCT) {
            
            redshift_int_fcollz = (int)floor( ( REDSHIFT_SAMPLE - determine_zpp_min )/zpp_bin_width );
            redshift_table_fcollz_diff = ( REDSHIFT_SAMPLE - determine_zpp_min - zpp_bin_width*(float)redshift_int_fcollz ) / zpp_bin_width;

#ifdef MINI_HALO
            mean_f_coll_st = prev_mean_f_coll_st + FgtrM_st_SFR(dicke(REDSHIFT_SAMPLE), Mmin, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10, Mlim_Fstar, Mlim_Fesc);
            mean_f_coll_st_MINI = prev_mean_f_coll_st_MINI + FgtrM_st_SFR_MINI(dicke(REDSHIFT_SAMPLE), Mmin_MINI, Mcrit_atom, ALPHA_STAR, F_STAR10_MINI, Mlim_Fstar_MINI);

            // to do the CONTEMPORANEOUS_DUTYCYCLE, we need to calculate the prev_mean_... with the current Mturns
            if(sample_index > 0) 
            {
                mean_f_coll_st -= FgtrM_st_SFR(dicke(PREV_REDSHIFT), Mmin, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10, Mlim_Fstar, Mlim_Fesc);
                mean_f_coll_st_MINI -= FgtrM_st_SFR_MINI(dicke(PREV_REDSHIFT), Mmin_MINI, Mcrit_atom, ALPHA_STAR, F_STAR10_MINI, Mlim_Fstar_MINI);
            }

            // record into the prev_mean to do CONTEMPORANEOUS_DUTYCYCLE at next snapshot
            prev_mean_f_coll_st = mean_f_coll_st;
            prev_mean_f_coll_st_MINI = mean_f_coll_st_MINI;

            // below is to calculate the minimum f_coll following v1.4
            //f_coll_min = FgtrM_st_SFR(dicke(Z_HEAT_MAX), Mmin, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10, Mlim_Fstar, Mlim_Fesc);
            //f_coll_min_MINI = FgtrM_st_SFR_MINI(dicke(Z_HEAT_MAX), Mmin_MINI, Mcrit_atom, ALPHA_STAR, F_STAR10_MINI, Mlim_Fstar_MINI);
#else
            mean_f_coll_st = Fcollz_val[redshift_int_fcollz] + redshift_table_fcollz_diff *( Fcollz_val[redshift_int_fcollz+1] - Fcollz_val[redshift_int_fcollz] );
            
            //redshift_int_fcollz = (int)floor( ( Z_HEAT_MAX - determine_zpp_min )/zpp_bin_width );
            
            //redshift_table_fcollz_diff = ( Z_HEAT_MAX - determine_zpp_min - zpp_bin_width*(float)redshift_int_fcollz ) / zpp_bin_width;
            
            //f_coll_min = Fcollz_val[redshift_int_fcollz] + redshift_table_fcollz_diff *( Fcollz_val[redshift_int_fcollz+1] - Fcollz_val[redshift_int_fcollz] );
#endif
            }
        else {
            mean_f_coll_st = FgtrM_st_SFR(growth_factor,M_TURN,ALPHA_STAR,ALPHA_ESC,F_STAR10,F_ESC10,Mlim_Fstar,Mlim_Fesc);
        }
    }
    else {
        mean_f_coll_st = FgtrM_st(REDSHIFT_SAMPLE, M_MIN);
    }
    
#ifdef MINI_HALO
    if (mean_f_coll_st*ION_EFF_FACTOR + mean_f_coll_st_MINI*ION_EFF_FACTOR_MINI< HII_ROUND_ERR) // way too small to ionize anything...
#else
    if (mean_f_coll_st*ION_EFF_FACTOR < HII_ROUND_ERR) // way too small to ionize anything...
#endif
    {
        //printf( "The ST mean collapse fraction is %e, which is much smaller than the effective critical collapse fraction of %e\n I will just declare everything to be neutral\n", mean_f_coll_st, f_coll_crit);

        // find the neutral fraction
        if(USE_TS_FLUCT) {
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                xH[ct] = 1.-x_e_z[ct]; // convert from x_e to xH
                global_xH += xH[ct];
            }
            global_xH /= (double)HII_TOT_NUM_PIXELS;
        }
        else {
            init_heat();
            global_xH = 1. - xion_RECFAST(REDSHIFT_SAMPLE, 0);
            destruct_heat();
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                xH[ct] = global_xH;
            }
        }
        skip_deallocate = 1;

    }
    else {
        
        // Take the ionisation fraction from the X-ray ionisations from Ts.c (only if the calculate spin temperature flag is set)
        if(USE_TS_FLUCT) {
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        *((float *)xe_unfiltered + HII_R_FFT_INDEX(i,j,k)) = x_e_z[HII_R_INDEX(i,j,k)];
                    }
                }
            }
        }
    
        if(USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"../FFTW_Wisdoms/real_to_complex_%d.fftwf_wisdom",HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                memcpy(deltax_unfiltered, deltax_unfiltered_original, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deltax_unfiltered, (fftwf_complex *)deltax_unfiltered, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }

#ifdef MINI_HALO
        if(USE_FFTW_WISDOM) {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mmin_unfiltered, (fftwf_complex *)log10_Mmin_unfiltered, FFTW_WISDOM_ONLY);
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mmin_unfiltered, (fftwf_complex *)log10_Mmin_unfiltered, FFTW_ESTIMATE);
        }

        if(USE_FFTW_WISDOM) {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mmin_MINI_unfiltered, (fftwf_complex *)log10_Mmin_MINI_unfiltered, FFTW_WISDOM_ONLY);
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)log10_Mmin_MINI_unfiltered, (fftwf_complex *)log10_Mmin_MINI_unfiltered, FFTW_ESTIMATE);
        }
#endif
        
        
        if(USE_TS_FLUCT) {
            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)xe_unfiltered, (fftwf_complex *)xe_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)xe_unfiltered, (fftwf_complex *)xe_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
        }
        
        if (INHOMO_RECO){
            
            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)N_rec_unfiltered, (fftwf_complex *)N_rec_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)N_rec_unfiltered, (fftwf_complex *)N_rec_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
        }
        
//        fftwf_destroy_plan(plan);
        
        // remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from
        //  real space to k-space
        // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
     
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            deltax_unfiltered[ct] /= (HII_TOT_NUM_PIXELS+0.0);
        }
    
#ifdef MINI_HALO
        for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
            log10_Mmin_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
            log10_Mmin_MINI_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
        }
#endif

        if(USE_TS_FLUCT) {
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                xe_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
            }
        }
        
        if (INHOMO_RECO){
            for (ct=0; ct<HII_KSPACE_NUM_PIXELS; ct++){
                N_rec_unfiltered[ct] /= (double)HII_TOT_NUM_PIXELS;
            }
        }
        
        
        /*************************************************************************************/
        /***************** LOOP THROUGH THE FILTER RADII (in Mpc)  ***************************/
        /*************************************************************************************/
        // set the max radius we will use, making sure we are always sampling the same values of radius
        // (this avoids aliasing differences w redshift)
        
        int determine_R_intermediate;
        
        determine_R_intermediate = 0;
        
        short_completely_ionised = 0;
        // loop through the filter radii (in Mpc)
        erfc_denom_cell=1; //dummy value
        
        R=fmax(R_BUBBLE_MIN, (cell_length_factor*BOX_LEN/(float)HII_DIM));
        
        int N_RSTEPS, counter_R;
        
        counter = 0;
        while ((R - fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN)) <= FRACT_FLOAT_ERR ) {
            R*= DELTA_R_HII_FACTOR;
            determine_R_intermediate += 1;
            if(R >= fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN)) {
                stored_R = R/DELTA_R_HII_FACTOR;
            }
            counter += 1;
        }

        N_RSTEPS = counter;
        
        counter_R = N_RSTEPS;
        
        determine_R_intermediate = determine_R_intermediate - 2;
        
        R=fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN);
        
        LAST_FILTER_STEP = 0;
     
        // YQ: I don't think this line has any meaning, but whatever...
        initialiseSplinedSigmaM(M_MIN,1e16);
        
        first_step_R = 1;
        
        counter = 0;

        while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){
            
            // Check if we are the last filter step
            if ( ((R/DELTA_R_HII_FACTOR - cell_length_factor*BOX_LEN/(float)HII_DIM) <= FRACT_FLOAT_ERR) || ((R/DELTA_R_HII_FACTOR - R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
                LAST_FILTER_STEP = 1;
                R = fmax(cell_length_factor*BOX_LEN/(double)HII_DIM, R_BUBBLE_MIN);
            }
            
            // Copy all relevant quantities from memory into new arrays to be smoothed and FFT'd.
            if(USE_TS_FLUCT) {
                memcpy(xe_filtered, xe_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }
            if (INHOMO_RECO){
                memcpy(N_rec_filtered, N_rec_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            }
            memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
#ifdef MINI_HALO
            memcpy(log10_Mmin_filtered, log10_Mmin_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            memcpy(log10_Mmin_MINI_filtered, log10_Mmin_MINI_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
#endif
            
            if (!LAST_FILTER_STEP || (R > cell_length_factor*BOX_LEN/(double)HII_DIM) ){
                if(USE_TS_FLUCT) {
                    HII_filter(xe_filtered, HII_FILTER, R);
                }
                if (INHOMO_RECO){
                    HII_filter(N_rec_filtered, HII_FILTER, R);
                }
                HII_filter(deltax_filtered, HII_FILTER, R);
#ifdef MINI_HALO
                HII_filter(log10_Mmin_filtered, HII_FILTER, R);
                HII_filter(log10_Mmin_MINI_filtered, HII_FILTER, R);
#endif
            }
            
            // Perform FFTs
            if(USE_FFTW_WISDOM) {
                // Check to see if wisdom exists, if not create it
                sprintf(wisdom_filename,"../FFTW_Wisdoms/complex_to_real_%d.fftwf_wisdom",HII_DIM);
                if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
                else {
                    if(first_step_R) {
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_PATIENT);
                        fftwf_execute(plan);
                        
                        // Store the wisdom for later use
                        fftwf_export_wisdom_to_filename(wisdom_filename);
                        
                        memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                        
                        HII_filter(deltax_filtered, HII_FILTER, R);
                        
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                    else {
                        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_WISDOM_ONLY);
                        fftwf_execute(plan);
                    }
                }
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_filtered, (float *)deltax_filtered, FFTW_ESTIMATE);
                fftwf_execute(plan);
            }
//            fftwf_destroy_plan(plan);
            
            if (USE_TS_FLUCT) {
                if(USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)xe_filtered, (float *)xe_filtered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)xe_filtered, (float *)xe_filtered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);
            }
            
            if (INHOMO_RECO){
                if(USE_FFTW_WISDOM) {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_filtered, (float *)N_rec_filtered, FFTW_WISDOM_ONLY);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_filtered, (float *)N_rec_filtered, FFTW_ESTIMATE);
                }
                fftwf_execute(plan);
            }
#ifdef MINI_HALO
            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mmin_filtered, (float *)log10_Mmin_filtered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mmin_filtered, (float *)log10_Mmin_filtered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);

            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mmin_MINI_filtered, (float *)log10_Mmin_MINI_filtered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)log10_Mmin_MINI_filtered, (float *)log10_Mmin_MINI_filtered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
#endif
            
            // Check if this is the last filtering scale.  If so, we don't need deltax_unfiltered anymore.
            // We will re-read it to get the real-space field, which we will use to set the residual neutral fraction
            ST_over_PS_HII = 0;
            f_coll = 0;
#ifdef MINI_HALO
            ST_over_PS_HII_MINI = 0;
            f_coll_MINI = 0;
#endif
            massofscaleR = RtoM(R);
            
            
            min_density = max_density = 0.0;
            
            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<HII_DIM; z++){
                        // delta cannot be less than -1
                        *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);
                        
                        if( *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) < min_density ) {
                            min_density = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                        }
                        if( *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) > max_density ) {
                            max_density = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
                        }
#ifdef MINI_HALO
                        // M_MINa cannot be less than Mcrit_atom
                        if (*((float *)log10_Mmin_filtered + HII_R_FFT_INDEX(x,y,z)) < log10_Mcrit_atom)
                          *((float *)log10_Mmin_filtered + HII_R_FFT_INDEX(x,y,z)) = log10_Mcrit_atom;
                        if (*((float *)log10_Mmin_filtered + HII_R_FFT_INDEX(x,y,z)) > LOG10MTURN_MAX)
                          *((float *)log10_Mmin_filtered + HII_R_FFT_INDEX(x,y,z)) = LOG10MTURN_MAX;
                        // M_MINa cannot be less than Mcrit_mol
                        if (*((float *)log10_Mmin_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) < log10_Mcrit_mol)
                          *((float *)log10_Mmin_MINI_filtered + HII_R_FFT_INDEX(x,y,z))  = log10_Mcrit_mol;
                          if (*((float *)log10_Mmin_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) > LOG10MTURN_MAX)
                              *((float *)log10_Mmin_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) = LOG10MTURN_MAX;
#endif
                    }
                }
            }
            
            if(HII_FILTER==1) {
                if((0.413566994*R*2.*PI/BOX_LEN) > 1.) {
                    // The sharp k-space filter will set every cell to zero, and the interpolation table using a flexible min/max density will fail.
                    
                    min_density = -1. + 9e-8;
                    max_density = 1.5*1.001;
                }
            }
            
            overdense_small_min = log10(1. + min_density);
            if(max_density > 1.5*1.001) {
                overdense_small_bin_width = 1./((double)NSFR_low-1.)*(log10(1.+1.5*1.001)-overdense_small_min);
            }
            else {
                overdense_small_bin_width = 1./((double)NSFR_low-1.)*(log10(1.+max_density)-overdense_small_min);
            }
            overdense_small_bin_width_inv = 1./overdense_small_bin_width;
            
            
            // New in v1.4
            if(USE_MASS_DEPENDENT_ZETA) {
#ifdef MINI_HALO
                initialiseGL_FcollSFR(NGL_SFR, M_MIN,massofscaleR);
                initialiseFcollSFR_spline(REDSHIFT_SAMPLE,min_density,max_density,M_MIN,massofscaleR,Mturn_interp_table,ALPHA_STAR,ALPHA_ESC,F_STAR10,F_ESC10,Mlim_Fstar,Mlim_Fesc,F_STAR10_MINI,Mlim_Fstar_MINI);
#else
                initialiseGL_FcollSFR(NGL_SFR, M_TURN/50.,massofscaleR);
                initialiseFcollSFR_spline(REDSHIFT_SAMPLE,min_density,max_density,massofscaleR,M_TURN,ALPHA_STAR,ALPHA_ESC,F_STAR10,F_ESC10,Mlim_Fstar,Mlim_Fesc);
#endif
            }
            else {
                erfc_denom = 2.*(pow(sigma_z0(M_MIN), 2) - pow(sigma_z0(massofscaleR), 2) );
                if (erfc_denom < 0) { // our filtering scale has become too small
                    break;
                }
                erfc_denom = sqrt(erfc_denom);
                erfc_denom = 1./( growth_factor * erfc_denom );

            }
            
            if(!USE_FCOLL_IONISATION_TABLE) {
                
                // Determine the global averaged f_coll for the overall normalisation
            
                // renormalize the collapse fraction so that the mean matches ST,
                // since we are using the evolved (non-linear) density field
                for (x=0; x<HII_DIM; x++){
                    for (y=0; y<HII_DIM; y++){
                        for (z=0; z<HII_DIM; z++){
                            
                            // delta cannot be less than -1
                            *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)) , -1.+FRACT_FLOAT_ERR);
                            
                            // <N_rec> cannot be less than zero
                            if (INHOMO_RECO){
                                *((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)N_rec_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.0);
                            }
                        
                            // x_e has to be between zero and unity
                            if (USE_TS_IN_21CM){
                                *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMAX(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.);
                                *((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) = FMIN(*((float *)xe_filtered + HII_R_FFT_INDEX(x,y,z)) , 0.999);
                            }
                
                            curr_dens = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z));
#ifdef MINI_HALO
                            prev_dens = *((float *)deltax_prev_filtered[counter_R] + HII_R_FFT_INDEX(x,y,z));
                            log10_Mmin_val = ( *((float *)log10_Mmin_filtered + HII_R_FFT_INDEX(x,y,z)) - LOG10MTURN_MIN) / LOG10MTURN_INT;
                            log10_Mmin_int = (int)floorf( log10_Mmin_val );
                            log10_Mmin_diff = log10_Mmin_val - (float)log10_Mmin_int;
                            log10_Mmin_MINI_val = ( *((float *)log10_Mmin_MINI_filtered + HII_R_FFT_INDEX(x,y,z)) - LOG10MTURN_MIN) / LOG10MTURN_INT;
                            log10_Mmin_MINI_int = (int)floorf( log10_Mmin_MINI_val );
                            log10_Mmin_MINI_diff = log10_Mmin_MINI_val - (float)log10_Mmin_MINI_int;
#endif
                        
                           // New in v1.4
                            if(USE_MASS_DEPENDENT_ZETA) {
                                // Usage of 0.99*Deltac arises due to the fact that close to the critical density, the collapsed fraction becomes a little unstable
                                // However, such densities should always be collapsed, so just set f_coll to unity. Additionally, the fraction of points in this regime relative
                                // to the entire simulation volume is extremely small.
                                
                                // NOTE: Again, be careful how this quantity has been defined. There is no off-set like in the analogous X-ray SFR case, however, I have retained
                                // The exp (log(10) * exponent) for evaluating the interpolation tables (for computational efficiency).
                                // Thus in the exp(splined_Fcoll) evaluation, splined_Fcoll = log(10) * exponent.
                                
#ifdef MINI_HALO
                                if (prev_dens < 1.5){
                                    
                                    if (prev_dens < -1.) {
                                        prev_Splined_Fcoll = 0;
                                        prev_Splined_Fcoll_MINI = 0;
                                    }
                                    else {
                                        prev_dens_val = (log10f(prev_dens+1.) - prev_overdense_small_min[counter_R])*prev_overdense_small_bin_width_inv[counter_R];
                                        prev_overdense_int = (int)floorf( prev_dens_val );
                                        prev_dens_diff = prev_dens_val - (float)prev_overdense_int;

                                        index_left = prev_overdense_int+log10_Mmin_int*NSFR_low;
                                        index_right = prev_overdense_int+(log10_Mmin_int+1)*NSFR_low;
                                        prev_Splined_Fcoll_left = prev_log10_Fcoll_spline_SFR[counter_R][index_left]*(1.-prev_dens_diff) + prev_log10_Fcoll_spline_SFR[counter_R][index_left+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_right = prev_log10_Fcoll_spline_SFR[counter_R][index_right]*(1.-prev_dens_diff) + prev_log10_Fcoll_spline_SFR[counter_R][index_right+1]*prev_dens_diff;
                                        prev_Splined_Fcoll = prev_Splined_Fcoll_left * (1. - log10_Mmin_diff) + prev_Splined_Fcoll_right * log10_Mmin_diff;
                                        prev_Splined_Fcoll = expf(prev_Splined_Fcoll);
                                        
                                        index_left = prev_overdense_int+log10_Mmin_MINI_int*NSFR_low;
                                        index_right = prev_overdense_int+(log10_Mmin_MINI_int+1)*NSFR_low;
                                        prev_Splined_Fcoll_MINI_left = prev_log10_Fcoll_spline_SFR_MINI[counter_R][index_left]*(1.-prev_dens_diff) + prev_log10_Fcoll_spline_SFR_MINI[counter_R][index_left+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_MINI_right = prev_log10_Fcoll_spline_SFR_MINI[counter_R][index_right]*(1.-prev_dens_diff) + prev_log10_Fcoll_spline_SFR_MINI[counter_R][index_right+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_MINI = prev_Splined_Fcoll_MINI_left * (1. - log10_Mmin_MINI_diff) + prev_Splined_Fcoll_MINI_right * log10_Mmin_MINI_diff;
                                        prev_Splined_Fcoll_MINI = expf(prev_Splined_Fcoll_MINI);

                                    }
                                }
                                else {
                                    if (prev_dens < 0.9*Deltac) {
                                        
                                        prev_dens_val = (prev_dens - prev_overdense_large_min[counter_R])*prev_overdense_large_bin_width_inv[counter_R];
                                        prev_overdense_int = (int)floorf( prev_dens_val );
                                        prev_dens_diff = prev_dens_val - (float)prev_overdense_int;

                                        index_left = prev_overdense_int+log10_Mmin_int*NSFR_high;
                                        index_right = prev_overdense_int+(log10_Mmin_int+1)*NSFR_high;
                                        prev_Splined_Fcoll_left = prev_Fcoll_spline_SFR[counter_R][index_left]*(1.-prev_dens_diff) + prev_Fcoll_spline_SFR[counter_R][index_left+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_right = prev_Fcoll_spline_SFR[counter_R][index_right]*(1.-prev_dens_diff) + prev_Fcoll_spline_SFR[counter_R][index_right+1]*prev_dens_diff;
                                        prev_Splined_Fcoll = prev_Splined_Fcoll_left * (1. - log10_Mmin_diff) + prev_Splined_Fcoll_right * log10_Mmin_diff;

                                        index_left = prev_overdense_int+log10_Mmin_MINI_int*NSFR_high;
                                        index_right = prev_overdense_int+(log10_Mmin_MINI_int+1)*NSFR_high;
                                        prev_Splined_Fcoll_MINI_left = prev_Fcoll_spline_SFR_MINI[counter_R][index_left]*(1.-prev_dens_diff) + prev_Fcoll_spline_SFR_MINI[counter_R][index_left+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_MINI_right = prev_Fcoll_spline_SFR_MINI[counter_R][index_right]*(1.-prev_dens_diff) + prev_Fcoll_spline_SFR_MINI[counter_R][index_right+1]*prev_dens_diff;
                                        prev_Splined_Fcoll_MINI = prev_Splined_Fcoll_MINI_left * (1. - log10_Mmin_MINI_diff) + prev_Splined_Fcoll_MINI_right * log10_Mmin_MINI_diff;
                                    }
                                    else {
                                        prev_Splined_Fcoll = 1.;
                                        prev_Splined_Fcoll_MINI = 1.;
                                    }
                                }

                                if (curr_dens < 1.5){
                                    
                                    if (curr_dens < -1.) {
                                        Splined_Fcoll = 0;
                                        Splined_Fcoll_MINI = 0;
                                    }
                                    else {
                                        dens_val = (log10f(curr_dens+1.) - overdense_small_min)*overdense_small_bin_width_inv;
                                        overdense_int = (int)floorf( dens_val );
                                        dens_diff = dens_val - (float)overdense_int;

                                        index_left = overdense_int+log10_Mmin_int*NSFR_low;
                                        index_right = overdense_int+(log10_Mmin_int+1)*NSFR_low;
                                        Splined_Fcoll_left = log10_Fcoll_spline_SFR[index_left]*(1.-dens_diff) + log10_Fcoll_spline_SFR[index_left+1]*dens_diff;
                                        Splined_Fcoll_right = log10_Fcoll_spline_SFR[index_right]*(1.-dens_diff) + log10_Fcoll_spline_SFR[index_right+1]*dens_diff;
                                        Splined_Fcoll = Splined_Fcoll_left * (1. - log10_Mmin_diff) + Splined_Fcoll_right * log10_Mmin_diff;
                                        Splined_Fcoll = expf(Splined_Fcoll);
                                        
                                        index_left = overdense_int+log10_Mmin_MINI_int*NSFR_low;
                                        index_right = overdense_int+(log10_Mmin_MINI_int+1)*NSFR_low;
                                        Splined_Fcoll_MINI_left = log10_Fcoll_spline_SFR_MINI[index_left]*(1.-dens_diff) + log10_Fcoll_spline_SFR_MINI[index_left+1]*dens_diff;
                                        Splined_Fcoll_MINI_right = log10_Fcoll_spline_SFR_MINI[index_right]*(1.-dens_diff) + log10_Fcoll_spline_SFR_MINI[index_right+1]*dens_diff;
                                        Splined_Fcoll_MINI = Splined_Fcoll_MINI_left * (1. - log10_Mmin_MINI_diff) + Splined_Fcoll_MINI_right * log10_Mmin_MINI_diff;
                                        Splined_Fcoll_MINI = expf(Splined_Fcoll_MINI);
                                    }
                                }
                                else {
                                    if (curr_dens < 0.9*Deltac) {
                                        
                                        dens_val = (curr_dens - overdense_large_min)*overdense_large_bin_width_inv;
                                        overdense_int = (int)floorf( dens_val );
                                        dens_diff = dens_val - (float)overdense_int;

                                        index_left = overdense_int+log10_Mmin_int*NSFR_high;
                                        index_right = overdense_int+(log10_Mmin_int+1)*NSFR_high;
                                        Splined_Fcoll_left = Fcoll_spline_SFR[index_left]*(1.-dens_diff) + Fcoll_spline_SFR[index_left+1]*dens_diff;
                                        Splined_Fcoll_right = Fcoll_spline_SFR[index_right]*(1.-dens_diff) + Fcoll_spline_SFR[index_right+1]*dens_diff;
                                        Splined_Fcoll = Splined_Fcoll_left * (1. - log10_Mmin_diff) + Splined_Fcoll_right * log10_Mmin_diff;

                                        index_left = overdense_int+log10_Mmin_MINI_int*NSFR_high;
                                        index_right = overdense_int+(log10_Mmin_MINI_int+1)*NSFR_high;
                                        Splined_Fcoll_MINI_left = Fcoll_spline_SFR_MINI[index_left]*(1.-dens_diff) + Fcoll_spline_SFR_MINI[index_left+1]*dens_diff;
                                        Splined_Fcoll_MINI_right = Fcoll_spline_SFR_MINI[index_right]*(1.-dens_diff) + Fcoll_spline_SFR_MINI[index_right+1]*dens_diff;
                                        Splined_Fcoll_MINI = Splined_Fcoll_MINI_left * (1. - log10_Mmin_MINI_diff) + Splined_Fcoll_MINI_right * log10_Mmin_MINI_diff;
                                    }
                                    else {
                                        Splined_Fcoll = 1.;
                                        Splined_Fcoll_MINI = 1.;
                                    }
                                }
#else
                                if (curr_dens < 1.5){
                                    
                                    if (curr_dens < -1.) {
                                        Splined_Fcoll = 0;
                                    }
                                    else {
                                        dens_val = (log10f(curr_dens+1.) - overdense_small_min)*overdense_small_bin_width_inv;
  
                                        overdense_int = (int)floorf( dens_val );
                                        
  
                                        Splined_Fcoll = log10_Fcoll_spline_SFR[overdense_int]*( 1 + (float)overdense_int - dens_val ) + log10_Fcoll_spline_SFR[overdense_int+1]*( dens_val - (float)overdense_int );
                                        
                                        Splined_Fcoll = expf(Splined_Fcoll);
                                        
                                    }
                                }
                                else {
                                    if (curr_dens < 0.9*Deltac) {
                                        
                                        dens_val = (curr_dens - overdense_large_min)*overdense_large_bin_width_inv;
                                        
                                        overdense_int = (int)floorf( dens_val );

                                        Splined_Fcoll = Fcoll_spline_SFR[overdense_int]*( 1 + (float)overdense_int - dens_val ) + Fcoll_spline_SFR[overdense_int+1]*( dens_val - (float)overdense_int );
                                    }
                                    else {
                                        Splined_Fcoll = 1.;
                                    }
                                }
#endif

                            }
                            else {
                            
                                erfc_arg_val = (Deltac - curr_dens)*erfc_denom;
                                if( erfc_arg_val < erfc_arg_min || erfc_arg_val > erfc_arg_max ) {
                                    Splined_Fcoll = splined_erfc(erfc_arg_val);
                                }
                                else {
                                    erfc_arg_val_index = (int)floor(( erfc_arg_val - erfc_arg_min )*InvArgBinWidth);
                                    Splined_Fcoll = ERFC_VALS[erfc_arg_val_index] + (erfc_arg_val - (erfc_arg_min + ArgBinWidth*(double)erfc_arg_val_index))*ERFC_VALS_DIFF[erfc_arg_val_index]*InvArgBinWidth;
                                }
                            }
                        
                            // save the value of the collasped fraction into the Fcoll array
#ifdef MINI_HALO
                            if (Splined_Fcoll > 1.) Splined_Fcoll = 1.;
                            if (Splined_Fcoll < 0.) Splined_Fcoll = 1e-40;
                            if (Splined_Fcoll_MINI > 1.) Splined_Fcoll_MINI = 1.;
                            if (Splined_Fcoll_MINI < 0.) Splined_Fcoll_MINI = 1e-40;

                            if (prev_Splined_Fcoll > 1.) prev_Splined_Fcoll = 1.;
                            if (prev_Splined_Fcoll < 0.) prev_Splined_Fcoll = 1e-40;
                            if (prev_Splined_Fcoll_MINI > 1.) prev_Splined_Fcoll_MINI = 1.;
                            if (prev_Splined_Fcoll_MINI < 0.) prev_Splined_Fcoll_MINI = 1e-40;

                            Fcoll[HII_R_INDEX(x,y,z)] = prev_Fcoll[counter_R][HII_R_INDEX(x,y,z)] + Splined_Fcoll - prev_Splined_Fcoll;
                            Fcoll_MINI[HII_R_INDEX(x,y,z)] = prev_Fcoll_MINI[counter_R][HII_R_INDEX(x,y,z)] + Splined_Fcoll_MINI - prev_Splined_Fcoll_MINI;

                            if (Fcoll[HII_R_INDEX(x,y,z)] > 1.) Fcoll[HII_R_INDEX(x,y,z)] = 1.;
                            if (Fcoll_MINI[HII_R_INDEX(x,y,z)] > 1.) Fcoll_MINI[HII_R_INDEX(x,y,z)] = 1.;

                            prev_Fcoll[counter_R][HII_R_INDEX(x,y,z)] = Fcoll[HII_R_INDEX(x,y,z)];
                            prev_Fcoll_MINI[counter_R][HII_R_INDEX(x,y,z)] = Fcoll_MINI[HII_R_INDEX(x,y,z)];

                            f_coll      += Fcoll[HII_R_INDEX(x,y,z)];
                            f_coll_MINI += Fcoll_MINI[HII_R_INDEX(x,y,z)];
#else
                            Fcoll[HII_R_INDEX(x,y,z)] = Splined_Fcoll;
                            f_coll += Splined_Fcoll;
#endif
                        }
                    }
                } //  end loop through Fcoll box
            
                f_coll /= (double) HII_TOT_NUM_PIXELS;
                // To avoid ST_over_PS becoms nan when f_coll = 0, I set f_coll = FRACT_FLOAT_ERR.
                //if (f_coll <= FRACT_FLOAT_ERR) f_coll = FRACT_FLOAT_ERR;
                //if (f_coll <= f_coll_min) f_coll = f_coll_min;
#ifdef MINI_HALO
                f_coll_MINI /= (double) HII_TOT_NUM_PIXELS;
                //if (f_coll_MINI <= f_coll_min_MINI) f_coll_MINI = f_coll_min_MINI;
#endif
            }
            else {
                
                // Evaluate the interpolation table of the global average f_coll
                
                R_MFP_INT_1 = (int)floor((R - R_MFP_MIN)/R_MFP_BINWIDTH);
                R_MFP_INT_2 = R_MFP_INT_1 + 1;
                
                R_MFP_VAL_1 = R_MFP_MIN + (R_MFP_UB - R_MFP_MIN)*(float)R_MFP_INT_1/((float)R_MFP_STEPS - 1.);
                R_MFP_VAL_2 = R_MFP_MIN + (R_MFP_UB - R_MFP_MIN)*(float)R_MFP_INT_2/((float)R_MFP_STEPS - 1.);
                
                if (LAST_FILTER_STEP){
                    f_coll = Ionisation_fcoll_table_final[TVIR_INT_1 + TVIR_STEPS*sample_index] + ( log10(ION_Tvir_MIN) - TVIR_VAL_1 )*( Ionisation_fcoll_table_final[TVIR_INT_2 + TVIR_STEPS*sample_index] - Ionisation_fcoll_table_final[TVIR_INT_1 + TVIR_STEPS*sample_index] )/( TVIR_VAL_2 - TVIR_VAL_1 );
                }
                else {
                    
                    f_coll_from_table_1 = ( R_MFP_VAL_2 - R )*Ionisation_fcoll_table[TVIR_INT_1 + TVIR_STEPS*( R_MFP_INT_1 + R_MFP_STEPS*sample_index )];
                    f_coll_from_table_1 += ( R - R_MFP_VAL_1 )*Ionisation_fcoll_table[TVIR_INT_1 + TVIR_STEPS*( R_MFP_INT_2 + R_MFP_STEPS*sample_index )];
                    f_coll_from_table_1 /= ( R_MFP_VAL_2 - R_MFP_VAL_1 );
                    
                    f_coll_from_table_2 = ( R_MFP_VAL_2 - R )*Ionisation_fcoll_table[TVIR_INT_2 + TVIR_STEPS*( R_MFP_INT_1 + R_MFP_STEPS*sample_index )];
                    f_coll_from_table_2 += ( R - R_MFP_VAL_1 )*Ionisation_fcoll_table[TVIR_INT_2 + TVIR_STEPS*( R_MFP_INT_2 + R_MFP_STEPS*sample_index )];
                    f_coll_from_table_2 /= ( R_MFP_VAL_2 - R_MFP_VAL_1 );
                    
                    f_coll = ( ( TVIR_VAL_2 - log10(ION_Tvir_MIN) )*f_coll_from_table_1 + ( log10(ION_Tvir_MIN) - TVIR_VAL_1 )*f_coll_from_table_2 )/( TVIR_VAL_2 - TVIR_VAL_1 );
                }
            }
            
            ST_over_PS_HII = mean_f_coll_st/f_coll;
#ifdef MINI_HALO
            ST_over_PS_HII_MINI = mean_f_coll_st_MINI/f_coll_MINI;
            //printf("ST_over_PS_HII=%g, ST_over_PS_HII_MINI=%g, mean_f_coll_st=%g, mean_f_coll_st_MINI=%g, f_coll=%g, f_coll_MINI=%g\n",ST_over_PS_HII, ST_over_PS_HII_MINI, mean_f_coll_st, mean_f_coll_st_MINI, f_coll, f_coll_MINI);
#endif
            
            //////////////////////////////  MAIN LOOP THROUGH THE BOX ///////////////////////////////////
            // now lets scroll through the filtered box
            
            rec = 0.;
            
            xHI_from_xrays = 1;
            //Gamma_R_prefactor = pow(1.+REDSHIFT_SAMPLE, 2) * (R*CMperMPC) * SIGMA_HI * ALPHA_UVB / (ALPHA_UVB+2.75) * N_b0 * HII_EFF_FACTOR / 1.0e-12;
            Gamma_R_prefactor = pow(1.+REDSHIFT_SAMPLE, 2) * (R*CMperMPC) * SIGMA_HI * ALPHA_UVB / (ALPHA_UVB+2.75) * N_b0 * ION_EFF_FACTOR / 1.0e-12;
            
            Gamma_R_prefactor /= t_ast;
#ifdef MINI_HALO
            Gamma_R_prefactor_MINI = Gamma_R_prefactor / ION_EFF_FACTOR * ION_EFF_FACTOR_MINI;
#endif
            
            if(!USE_FCOLL_IONISATION_TABLE) {
                LOOP_INDEX = HII_DIM;
            }
            else {
                LOOP_INDEX = total_in_z;
            }

            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<LOOP_INDEX; z++){
                        
                        if(USE_FCOLL_IONISATION_TABLE) {
                            if((min_slice_index + z) >= HII_DIM) {
                                slice_index_reducedLC = (min_slice_index + z) - HII_DIM;
                            }
                            else {
                                slice_index_reducedLC = (min_slice_index + z);
                            }
                            
                            // delta cannot be less than -1
                            *((float *)deltax_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) = FMAX(*((float *)deltax_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) , -1.+FRACT_FLOAT_ERR);
                            
                            // <N_rec> cannot be less than zero
                            if (INHOMO_RECO){
                                *((float *)N_rec_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) = FMAX(*((float *)N_rec_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) , 0.0);
                            }
                            
                            // x_e has to be between zero and unity
                            if (USE_TS_IN_21CM){
                                *((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) = FMAX(*((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) , 0.);
                                *((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) = FMIN(*((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)) , 0.999);
                            }
                        }
                        else {
                            slice_index_reducedLC = z;
                        }
                        
                        curr_dens = *((float *)deltax_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC));
                        
                        if(USE_FCOLL_IONISATION_TABLE) {
                            // New in v1.4: current version do not support to use this option for the mass dependent ionizing efficiency.
                            if(USE_MASS_DEPENDENT_ZETA) {
                                
                                if(curr_dens < 0.9*Deltac) {
                                    // This is here as the interpolation tables have some issues very close
                                    // to Deltac. So lets just assume these voxels collapse anyway.
                                    FcollSpline_SFR(curr_dens,&(Splined_Fcoll));
                                    // Using the fcoll ionisation table, fcoll is not computed in each cell.
                                    // For this option, fcoll must be calculated.
                                }
                                else {
                                    Splined_Fcoll = 1.;
                                }
                                
                            }
                            // check for aliasing which can occur for small R and small cell sizes,
                            // since we are using the analytic form of the window function for speed and simplicity
                            else {

                                erfc_arg_val = (Deltac - curr_dens)*erfc_denom;
                                if( erfc_arg_val < erfc_arg_min || erfc_arg_val > erfc_arg_max ) {
                                    Splined_Fcoll = splined_erfc(erfc_arg_val);
                                }
                                else {
                                    erfc_arg_val_index = (int)floor(( erfc_arg_val - erfc_arg_min )*InvArgBinWidth);
                                    Splined_Fcoll = ERFC_VALS[erfc_arg_val_index] + (erfc_arg_val - (erfc_arg_min + ArgBinWidth*(double)erfc_arg_val_index))*ERFC_VALS_DIFF[erfc_arg_val_index]*InvArgBinWidth;
                                }
                            }
                        }
                        else {
                            
                            Splined_Fcoll = Fcoll[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)];
#ifdef MINI_HALO
                            Splined_Fcoll_MINI = Fcoll_MINI[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)];
#endif
                        }
                        
                        f_coll = ST_over_PS_HII * Splined_Fcoll;
                        //if (f_coll <= f_coll_min) f_coll = f_coll_min;
#ifdef MINI_HALO
                        f_coll_MINI = ST_over_PS_HII_MINI * Splined_Fcoll_MINI;
                        //if (f_coll_MINI <= f_coll_min_MINI) f_coll_MINI = f_coll_min_MINI;
#endif

                        
                        if (INHOMO_RECO){
//                            dfcolldt = f_coll / t_ast;
//                            Gamma_R = Gamma_R_prefactor * dfcolldt;

                            rec = (*((float *)N_rec_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC))); // number of recombinations per mean baryon
                            rec /= (1. + curr_dens); // number of recombinations per baryon inside <R>
                        }
                        
                        // adjust the denominator of the collapse fraction for the residual electron fraction in the neutral medium
                        if (USE_TS_IN_21CM){
                            xHI_from_xrays = (1. - *((float *)xe_filtered + coeval_box_pos_FFT(Default_LOS_direction,x,y,slice_index_reducedLC)));
                        }
                        
                        // check if fully ionized!
#ifdef MINI_HALO
                        if ( (f_coll*ION_EFF_FACTOR + f_coll_MINI*ION_EFF_FACTOR_MINI> xHI_from_xrays*(1.0+rec)) )
#else
                        if ( (f_coll*ION_EFF_FACTOR > xHI_from_xrays*(1.0+rec)) )
#endif
                        { //IONIZED!!
                            
                            // if this is the first crossing of the ionization barrier for this cell (largest R), record the gamma
                            // this assumes photon-starved growth of HII regions...  breaks down post EoR
                            if (INHOMO_RECO && (xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] > FRACT_FLOAT_ERR) ){
//                                Gamma12[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = Gamma_R;
                                Gamma12[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = Gamma_R_prefactor * f_coll;
#ifdef MINI_HALO
                                Gamma12[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] += Gamma_R_prefactor_MINI * f_coll_MINI;
#endif
                            }
                            
                            // keep track of the first time this cell is ionized (earliest time)
                            if (INHOMO_RECO && (z_re[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] < 0)){
                                z_re[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = REDSHIFT_SAMPLE;
                            }
                            
                            // FLAG CELL(S) AS IONIZED
                            if (FIND_BUBBLE_ALGORITHM == 2) // center method
                                xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = 0;
                            else if (FIND_BUBBLE_ALGORITHM == 1) // sphere method
                                update_in_sphere(xH, HII_DIM, R/BOX_LEN, x/(HII_DIM+0.0), y/(HII_DIM+0.0), slice_index_reducedLC/(HII_DIM+0.0));
                            else{
                                printf( "Incorrect choice of find bubble algorithm: %i\nAborting...", FIND_BUBBLE_ALGORITHM);
                                xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = 0;
                            }
                        } // end ionized
                        // If not fully ionized, then assign partial ionizations
                        else if (LAST_FILTER_STEP && (xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] > TINY)){
                                
                            if (f_coll>1) f_coll=1;
#ifdef MINI_HALO
                            if (f_coll_MINI>1) f_coll_MINI=1;
#endif
                            
#ifdef MINI_HALO
                            ave_N_min_cell = ( f_coll + f_coll_MINI ) * pixel_mass*(1. + curr_dens) / M_MIN; // ave # of M_MIN halos in cell
#else
                            ave_N_min_cell = f_coll * pixel_mass*(1. + curr_dens) / M_MIN; // ave # of M_MIN halos in cell
#endif
                            
                            if (ave_N_min_cell < N_POISSON){
                                if (ave_N_min_cell < 0.2){
                                    f_coll=0;
#ifdef MINI_HALO
                                    f_coll_MINI=0;
#endif
                                }
                                else{
                                    N_min_cell = (int) gsl_ran_poisson(r, N_POISSON) * ave_N_min_cell / (float) N_POISSON;
#ifdef MINI_HALO
                                    f_coll = N_min_cell * M_MIN / (pixel_mass*(1. + curr_dens)) * ((f_coll / (f_coll + f_coll_MINI)));
                                    f_coll_MINI = N_min_cell * M_MIN / (pixel_mass*(1. + curr_dens)) - f_coll;
#else
                                    f_coll = N_min_cell * M_MIN / (pixel_mass*(1. + curr_dens));
#endif
                                }
                            }

                            res_xH = xHI_from_xrays - f_coll * ION_EFF_FACTOR;
#ifdef MINI_HALO
                            res_xH -= f_coll_MINI * ION_EFF_FACTOR_MINI;
#endif

                            // and make sure fraction doesn't blow up for underdense pixels
                            if (res_xH < 0)
                                res_xH = 0;
                            else if (res_xH > 1)
                                res_xH = 1;
                            
                            xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)] = res_xH;
                        } // end partial ionizations at last filtering step
                    } // k
                } // j
            } // i
            
            if(!USE_FCOLL_IONISATION_TABLE && !INHOMO_RECO) {
                
                global_step_xH = 0;
                for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                    global_step_xH += xH[ct];
                }
                global_step_xH /= (float)HII_TOT_NUM_PIXELS;

                if(global_step_xH==0.0) {
                    short_completely_ionised = 1;
                    break;
                }
            }

#ifdef MINI_HALO
            //copy deltax_filtered to deltax_prev_filtered
            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<HII_DIM; z++){
                        *((float *)deltax_prev_filtered[counter_R] + HII_R_FFT_INDEX(x,y,z)) = *((float *)deltax_filtered + HII_R_FFT_INDEX(x,y,z)); 
                    }
                }
            }
            prev_overdense_large_min[counter_R] = overdense_large_min;
            prev_overdense_small_min[counter_R] = overdense_small_min;
            prev_overdense_large_bin_width_inv[counter_R] = overdense_large_bin_width_inv;
            prev_overdense_small_bin_width_inv[counter_R] = overdense_small_bin_width_inv;
            // store the result into prev_... to do the CONTEMPORANEOUS_DUTYCYCLE
            for (j=0;j<LOG10MTURN_NUM;j++){
                for (i=0; i<NSFR_low; i++){
                    prev_log10_Fcoll_spline_SFR[counter_R][i+j*NSFR_low] = log10_Fcoll_spline_SFR[i+j*NSFR_low];
                    prev_log10_Fcoll_spline_SFR_MINI[counter_R][i+j*NSFR_low] = log10_Fcoll_spline_SFR_MINI[i+j*NSFR_low];
                }
                for(i=0;i<NSFR_high;i++) {
                    prev_Fcoll_spline_SFR[counter_R][i+j*NSFR_high] = Fcoll_spline_SFR[i+j*NSFR_high];
                    prev_Fcoll_spline_SFR_MINI[counter_R][i+j*NSFR_high] = Fcoll_spline_SFR_MINI[i+j*NSFR_high];
                }
            }
#endif
            
            if(first_step_R) {
                R = stored_R;
                first_step_R = 0;
            }
            else {
                R /= DELTA_R_HII_FACTOR;
            }
            counter_R -= 1;
         
        }
        if(!USE_FCOLL_IONISATION_TABLE) {
            // find the neutral fraction
            global_xH = 0;
     
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                global_xH += xH[ct];
            }
            global_xH /= (float)HII_TOT_NUM_PIXELS;
        }
        else {
            // Estimate the neutral fraction from the reduced box. Can be handy to have, but shouldn't be trusted for anything more as only a fraction of the co-eval box is being used
            global_xH = 0;
            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<total_in_z; z++){
                        
                        if((min_slice_index + z) >= HII_DIM) {
                            slice_index_reducedLC = (min_slice_index + z) - HII_DIM;
                        }
                        else {
                            slice_index_reducedLC = (min_slice_index + z);
                        }
                        global_xH += xH[coeval_box_pos(Default_LOS_direction,x,y,slice_index_reducedLC)];
                    }
                }
            }
            global_xH /= ((float)HII_DIM*(float)HII_DIM*(float)total_in_z);
        }
        
        // update the N_rec field
        if (INHOMO_RECO){
            
            //fft to get the real N_rec  and delta fields
            // Wisdoms will already exists, so do not need to search for them or create them (created earlier in this function if don't already exist)
            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_unfiltered, (float *)N_rec_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)N_rec_unfiltered, (float *)N_rec_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
            if(USE_FFTW_WISDOM) {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_unfiltered, (float *)deltax_unfiltered, FFTW_WISDOM_ONLY);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)deltax_unfiltered, (float *)deltax_unfiltered, FFTW_ESTIMATE);
            }
            fftwf_execute(plan);
//            fftwf_destroy_plan(plan);
        
            for (x=0; x<HII_DIM; x++){
                for (y=0; y<HII_DIM; y++){
                    for (z=0; z<HII_DIM; z++){
                        curr_dens = 1.0 + (*((float *)deltax_unfiltered + HII_R_FFT_INDEX(x,y,z)));
                        z_eff = (1.+REDSHIFT_SAMPLE) * pow(curr_dens, 1.0/3.0) - 1;
                        dNrec = splined_recombination_rate(z_eff, Gamma12[HII_R_INDEX(x,y,z)]) * fabs_dtdz * ZSTEP * (1 - xH[HII_R_INDEX(x,y,z)]);
                        *((float *)N_rec_unfiltered + HII_R_FFT_INDEX(x,y,z)) += dNrec;
                    }
                }
            }
        }
    }

    // deallocate
    gsl_rng_free (r);
    
    nf = global_xH;
    if(STORE_DATA) {
        aveNF[sample_index] = nf;
    }
    
    if(!USE_LIGHTCONE) {
        if(PRINT_FILES) {
            sprintf(filename, "NeutralFraction_%f_%f_%f.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2,REDSHIFT_SAMPLE);
            F=fopen(filename, "wt");
            fprintf(F, "%lf\n",nf);
            fclose(F);
        }
    }
    ///////////////////////////////// End of perform 'find_HII_bubbles.c' ///////////////////////////////
    
    ///////////////////////////////////// Perform 'delta_T.c' ///////////////////////////////////////////
    
    float fz1, fz2, fz, z_slice;
    double dvdx, ave, max_v_deriv;
    float k_x, k_y, k_z, k_mag, const_factor, T_rad, pixel_Ts_factor, pixel_x_HI, pixel_deltax, H;
    
    ////////////////////////////////////  BEGIN INITIALIZATION //////////////////////////////////////////
    ave = 0;

    if(GenerateNewICs) {
            
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)v + HII_R_FFT_INDEX(i,j,k)) = LOWRES_velocity_REDSHIFT[HII_R_INDEX(i,j,k)];
                }
            }
        }
    }
    else {
        switch(VELOCITY_COMPONENT){
            case 1:  sprintf(filename, "../Boxes/updated_vx_z%06.2f_%i_%.0fMpc", REDSHIFT_SAMPLE, HII_DIM, BOX_LEN);
                break;
            case 3:  sprintf(filename, "../Boxes/updated_vz_z%06.2f_%i_%.0fMpc", REDSHIFT_SAMPLE, HII_DIM, BOX_LEN);
                break;
            default: sprintf(filename, "../Boxes/updated_vy_z%06.2f_%i_%.0fMpc", REDSHIFT_SAMPLE, HII_DIM, BOX_LEN);
        }
        F=fopen(filename, "rb");
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    if (fread((float *)v + HII_R_FFT_INDEX(i,j,k), sizeof(float), 1, F)!=1){
                        printf("Read error occured while reading velocity box.\n");
                        fclose(F);
                    }
                }
            }
        }
        fclose(F);
    }

    T_rad = T_cmb*(1.+REDSHIFT_SAMPLE);
    H = hubble(REDSHIFT_SAMPLE);
    const_factor = 27 * (OMb*hlittle*hlittle/0.023) *
    sqrt( (0.15/OMm/hlittle/hlittle) * (1.+REDSHIFT_SAMPLE)/10.0 );
    
    memcpy(deltax, deltax_unfiltered_original, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    ///////////////////////////////  END INITIALIZATION /////////////////////////////////////////////
    
    // ok, lets fill the delta_T box; which will be the same size as the bubble box

    for (i=0; i<HII_DIM; i++){
        for (j=0; j<HII_DIM; j++){
            for (k=0; k<HII_DIM; k++){
                
                pixel_deltax = deltax[HII_R_FFT_INDEX(i,j,k)];
                pixel_x_HI = xH[HII_R_INDEX(i,j,k)];
                
                delta_T[HII_R_INDEX(i,j,k)] = const_factor*pixel_x_HI*(1.+pixel_deltax);

                if (USE_TS_FLUCT) {
                        
                    if(SUBCELL_RSD) {
                        // Converting the prefactors into the optical depth, tau. Factor of 1000 is the conversion of spin temperature from K to mK
                        delta_T[HII_R_INDEX(i,j,k)] *= (1. + REDSHIFT_SAMPLE)/(1000.*Ts_z[HII_R_INDEX(i,j,k)]);
                    }
                    else {
                        pixel_Ts_factor = (1 - T_rad / Ts_z[HII_R_INDEX(i,j,k)]);
                        delta_T[HII_R_INDEX(i,j,k)] *= pixel_Ts_factor;
                    }
                }
                ave += delta_T[HII_R_INDEX(i,j,k)];
            }
        }
    }
    ave /= (float)HII_TOT_NUM_PIXELS;

    x_val1 = 0.;
    x_val2 = 1.;
        
    subcell_width = (BOX_LEN/(float)HII_DIM)/(float)N_RSD_STEPS;
        
    float max_cell_distance;
        
    max_cell_distance = 0.;

    // now write out the delta_T box
    if (T_USE_VELOCITIES){
        ave = 0.;

        memcpy(vel_gradient, v, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        
        // Wisdoms will already exists, so do not need to search for them or create them (created earlier in this function if don't already exist)
        if(USE_FFTW_WISDOM) {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_WISDOM_ONLY);
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)vel_gradient, (fftwf_complex *)vel_gradient, FFTW_ESTIMATE);
        }
        fftwf_execute(plan);
//        fftwf_destroy_plan(plan);
        
        for (n_x=0; n_x<HII_DIM; n_x++){
            if (n_x>HII_MIDDLE)
                k_x =(n_x-HII_DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;
            
            for (n_y=0; n_y<HII_DIM; n_y++){
                if (n_y>HII_MIDDLE)
                    k_y =(n_y-HII_DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;
                
                for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                    k_z = n_z * DELTA_K;
                    
                    // take partial deriavative along the line of sight
                    switch(VELOCITY_COMPONENT){
                        case 1:
                            *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_x*I/(float)HII_TOT_NUM_PIXELS;
                            break;
                        case 3:
                            *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_z*I/(float)HII_TOT_NUM_PIXELS;
                            break;
                        default:
                            *((fftwf_complex *) vel_gradient + HII_C_INDEX(n_x,n_y,n_z)) *= k_y*I/(float)HII_TOT_NUM_PIXELS;
                    }
                }
            }
        }
        
        // Wisdoms will already exists, so do not need to search for them or create them (created earlier in this function if don't already exist)
        if(USE_FFTW_WISDOM) {
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_WISDOM_ONLY);
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)vel_gradient, (float *)vel_gradient, FFTW_ESTIMATE);
        }
        fftwf_execute(plan);
//        fftwf_destroy_plan(plan);

        if(SUBCELL_RSD) {
            
            // now add the velocity correction to the delta_T maps
            min_gradient_component = 1.0;
            
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                        
                    if(!USE_FCOLL_IONISATION_TABLE) {
                        LOOP_INDEX = HII_DIM;
                    }
                    else {
                        LOOP_INDEX = total_in_z;
                    }
                    
                    for (k=0; k<LOOP_INDEX; k++){
                        
                        if(!USE_FCOLL_IONISATION_TABLE) {
                            slice_index_reducedLC = k;
                        }
                        else {
                            // We are using the reduced box, with padding, so need to only fill the arrays in the necessary cells
                            if((min_slice_index + k) >= HII_DIM) {
                                slice_index_reducedLC = (min_slice_index + k) - HII_DIM;
                            }
                            else {
                                slice_index_reducedLC = (min_slice_index + k);
                            }
                        }
                        
                        gradient_component = fabs(vel_gradient[HII_R_FFT_INDEX(i,j,slice_index_reducedLC)]/H + 1.0);
                                
                        // Calculate the brightness temperature, using the optical depth
                        if(gradient_component < FRACT_FLOAT_ERR) {
                            // Gradient component goes to zero, optical depth diverges. But, since we take exp(-tau), this goes to zero and (1 - exp(-tau)) goes to unity.
                            // Again, factors of 1000. are conversions from K to mK
                            delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)] = 1000.*(Ts_z[HII_R_INDEX(i,j,slice_index_reducedLC)] - T_rad)/(1. + REDSHIFT_SAMPLE);
                        }
                        else {
                            delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)] = (1. - exp(- delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/gradient_component ))*1000.*(Ts_z[HII_R_INDEX(i,j,slice_index_reducedLC)] - T_rad)/(1. + REDSHIFT_SAMPLE);
                        }
                    }
                }
            }
            
            // normalised units of cell length. 0 equals beginning of cell, 1 equals end of cell
            // These are the sub-cell central positions (x_pos_offset), and the corresponding normalised value (x_pos) between 0 and 1
            for(ii=0;ii<N_RSD_STEPS;ii++) {
                x_pos_offset[ii] = subcell_width*(float)ii + subcell_width/2.;
                x_pos[ii] = x_pos_offset[ii]/( BOX_LEN/(float)HII_DIM );
            }
            
            // Note to convert the velocity v, to a displacement in redshift space, convert from s -> r + (1.+z)*v/H(z)
            // To convert the velocity within the array v to km/s, it is a*dD/dt*delta. Where the scale factor a comes from the continuity equation
            // The array v as defined in 21cmFAST is (ik/k^2)*dD/dt*delta, as it is defined as a comoving quantity (scale factor is implicit).
            // However, the conversion between real and redshift space also picks up a scale factor, therefore the scale factors drop out and therefore
            // the displacement of the sub-cells is purely determined from the array, v and the Hubble factor: v/H.
            
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    
                    // Generate the optical-depth for the specific line-of-sight with R.S.D
                    for(k=0;k<HII_DIM;k++) {
                        delta_T_RSD_LOS[k] = 0.0;
                    }
                    
                    for (k=0; k<LOOP_INDEX; k++){
                    
                        if(!USE_FCOLL_IONISATION_TABLE) {
                            slice_index_reducedLC = k;
                        }
                        else {
                            if((min_slice_index + k) >= HII_DIM) {
                                slice_index_reducedLC = (min_slice_index + k) - HII_DIM;
                            }
                            else {
                                slice_index_reducedLC = (min_slice_index + k);
                            }
                        }
                        
                        if((fabs(delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]) >= FRACT_FLOAT_ERR) && (xH[HII_R_INDEX(i,j,slice_index_reducedLC)] >= FRACT_FLOAT_ERR)) {
                            
                            if(slice_index_reducedLC==0) {
                                d1_low = v[HII_R_FFT_INDEX(i,j,HII_DIM-1)]/H;
                                d2_low = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC)]/H;
                            }
                            else {
                                d1_low = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC-1)]/H;
                                d2_low = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC)]/H;
                            }
                            // Displacements (converted from velocity) for the original cell centres straddling half of the sub-cells (cell after)
                            if(slice_index_reducedLC==(HII_DIM-1)) {
                                d1_high = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC)]/H;
                                d2_high = v[HII_R_FFT_INDEX(i,j,0)]/H;
                            }
                            else {
                                d1_high = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC)]/H;
                                d2_high = v[HII_R_FFT_INDEX(i,j,slice_index_reducedLC+1)]/H;
                            }
                                    
                            for(ii=0;ii<N_RSD_STEPS;ii++) {
                                        
                                // linearly interpolate the displacements to determine the corresponding displacements of the sub-cells
                                // Checking of 0.5 is for determining if we are left or right of the mid-point of the original cell (for the linear interpolation of the displacement)
                                // to use the appropriate cell
                                        
                                if(x_pos[ii] <= 0.5) {
                                    subcell_displacement = d1_low + ( (x_pos[ii] + 0.5 ) - x_val1)*( d2_low - d1_low )/( x_val2 - x_val1 );
                                }
                                else {
                                    subcell_displacement = d1_high + ( (x_pos[ii] - 0.5 ) - x_val1)*( d2_high - d1_high )/( x_val2 - x_val1 );
                                }
                                        
                                // The new centre of the sub-cell post R.S.D displacement. Normalised to units of cell width for determining it's displacement
                                RSD_pos_new = (x_pos_offset[ii] + subcell_displacement)/( BOX_LEN/(float)HII_DIM );
                                // The sub-cell boundaries of the sub-cell, for determining the fractional contribution of the sub-cell to neighbouring cells when
                                // the sub-cell straddles two cell positions
                                RSD_pos_new_boundary_low = RSD_pos_new - (subcell_width/2.)/( BOX_LEN/(float)HII_DIM );
                                RSD_pos_new_boundary_high = RSD_pos_new + (subcell_width/2.)/( BOX_LEN/(float)HII_DIM );
                                        
                                if(RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_high < 1.0) {
                                    // sub-cell has remained in the original cell (just add it back to the original cell)
                                            
                                    delta_T_RSD_LOS[slice_index_reducedLC] += delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                }
                                else if(RSD_pos_new_boundary_low < 0.0 && RSD_pos_new_boundary_high < 0.0) {
                                    // sub-cell has moved completely into a new cell (toward the observer)
                                            
                                    // determine how far the sub-cell has moved in units of original cell boundary
                                    cell_distance = ceil(fabs(RSD_pos_new_boundary_low))-1.;
                                            
                                    // Determine the location of the sub-cell relative to the original cell binning
                                    if(fabs(RSD_pos_new_boundary_high) > cell_distance) {
                                        // sub-cell is entirely contained within the new cell (just add it to the new cell)
                                                
                                        // check if the new cell position is at the edge of the box. If so, periodic boundary conditions
                                        if(slice_index_reducedLC<((int)cell_distance+1)) {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance+1) + HII_DIM] += delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance+1)] += delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                    }
                                    else {
                                        // sub-cell is partially contained within the cell
                                                
                                        // Determine the fraction of the sub-cell which is in either of the two original cells
                                        fraction_outside = (fabs(RSD_pos_new_boundary_low) - cell_distance)/(subcell_width/( BOX_LEN/(float)HII_DIM ));
                                        fraction_within = 1. - fraction_outside;
                                                
                                        // Check if the first part of the sub-cell is at the box edge
                                        if(slice_index_reducedLC<(((int)cell_distance))) {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance) + HII_DIM] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance)] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        // Check if the second part of the sub-cell is at the box edge
                                        if(slice_index_reducedLC<(((int)cell_distance + 1))) {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance+1) + HII_DIM] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC-((int)cell_distance+1)] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                    }
                                }
                                else if(RSD_pos_new_boundary_low < 0.0 && (RSD_pos_new_boundary_high > 0.0 && RSD_pos_new_boundary_high < 1.0)) {
                                    // sub-cell has moved partially into a new cell (toward the observer)
                                            
                                    // Determine the fraction of the sub-cell which is in either of the two original cells
                                    fraction_within = RSD_pos_new_boundary_high/(subcell_width/( BOX_LEN/(float)HII_DIM ));
                                    fraction_outside = 1. - fraction_within;
                                            
                                    // Check the periodic boundaries conditions and move the fraction of each sub-cell to the appropriate new cell
                                    if(slice_index_reducedLC==0) {
                                        delta_T_RSD_LOS[HII_DIM-1] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        delta_T_RSD_LOS[slice_index_reducedLC] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                    }
                                    else {
                                        delta_T_RSD_LOS[slice_index_reducedLC-1] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        delta_T_RSD_LOS[slice_index_reducedLC] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                    }
                                }
                                else if((RSD_pos_new_boundary_low >= 0.0 && RSD_pos_new_boundary_low < 1.0) && (RSD_pos_new_boundary_high >= 1.0)) {
                                    // sub-cell has moved partially into a new cell (away from the observer)
                                            
                                    // Determine the fraction of the sub-cell which is in either of the two original cells
                                    fraction_outside = (RSD_pos_new_boundary_high - 1.)/(subcell_width/( BOX_LEN/(float)HII_DIM ));
                                    fraction_within = 1. - fraction_outside;
                                            
                                    // Check the periodic boundaries conditions and move the fraction of each sub-cell to the appropriate new cell
                                    if(slice_index_reducedLC==(HII_DIM-1)) {
                                        delta_T_RSD_LOS[slice_index_reducedLC] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        delta_T_RSD_LOS[0] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                    }
                                    else {
                                        delta_T_RSD_LOS[slice_index_reducedLC] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        delta_T_RSD_LOS[slice_index_reducedLC+1] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                    }
                                }
                                else {
                                    // sub-cell has moved completely into a new cell (away from the observer)
                                            
                                    // determine how far the sub-cell has moved in units of original cell boundary
                                    cell_distance = floor(fabs(RSD_pos_new_boundary_high));
                                        
                                    if(RSD_pos_new_boundary_low >= cell_distance) {
                                        // sub-cell is entirely contained within the new cell (just add it to the new cell)
                                                
                                        // check if the new cell position is at the edge of the box. If so, periodic boundary conditions
                                        if(slice_index_reducedLC>(HII_DIM - 1 - (int)cell_distance)) {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance - HII_DIM] += delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance] += delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                    }
                                    else {
                                        // sub-cell is partially contained within the cell
                                                
                                        // Determine the fraction of the sub-cell which is in either of the two original cells
                                        fraction_outside = (RSD_pos_new_boundary_high - cell_distance)/(subcell_width/( BOX_LEN/(float)HII_DIM ));
                                        fraction_within = 1. - fraction_outside;
                                                
                                        // Check if the first part of the sub-cell is at the box edge
                                        if(slice_index_reducedLC>(HII_DIM - 1 - ((int)cell_distance-1))) {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance-1 - HII_DIM] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance-1] += fraction_within*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        // Check if the second part of the sub-cell is at the box edge
                                        if(slice_index_reducedLC>(HII_DIM - 1 - ((int)cell_distance))) {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance - HII_DIM] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                        else {
                                            delta_T_RSD_LOS[slice_index_reducedLC+(int)cell_distance] += fraction_outside*delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)]/(float)N_RSD_STEPS;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if(!USE_FCOLL_IONISATION_TABLE) {
                        for(k=0;k<HII_DIM;k++) {
                            delta_T[HII_R_INDEX(i,j,k)] = delta_T_RSD_LOS[k];

                            ave += delta_T_RSD_LOS[k];
                        }
                    }
                    else {
                            
                        for(k=0;k<total_in_z;k++) {
                            
                            if((min_slice_index + k) >= HII_DIM) {
                                slice_index_reducedLC = (min_slice_index + k) - HII_DIM;
                            }
                            else {
                                slice_index_reducedLC = (min_slice_index + k);
                            }
                            delta_T[HII_R_INDEX(i,j,slice_index_reducedLC)] = delta_T_RSD_LOS[slice_index_reducedLC];
                            
                            ave += delta_T_RSD_LOS[slice_index_reducedLC];
                        }
                    }
                }
            }

            if(!USE_FCOLL_IONISATION_TABLE) {
                ave /= (float)HII_TOT_NUM_PIXELS;
            }
            else {
                ave /= ((float)HII_DIM*(float)HII_DIM*(float)total_in_z);
            }
        }
        else {
            
            // now add the velocity correction to the delta_T maps
            max_v_deriv = fabs(MAX_DVDR*H);
            
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                
                        dvdx = vel_gradient[HII_R_FFT_INDEX(i,j,k)];
                    
                        // set maximum allowed gradient for this linear approximation
                        if (fabs(dvdx) > max_v_deriv){
                            if (dvdx < 0) dvdx = -max_v_deriv;
                            else dvdx = max_v_deriv;
                            //                               nonlin_ct++;
                        }
                            
                        delta_T[HII_R_INDEX(i,j,k)] /= (dvdx/H + 1.0);

                        ave += delta_T[HII_R_INDEX(i,j,k)];
                    }
                }
            }
            ave /= (HII_TOT_NUM_PIXELS+0.0);
        }
    }

    // Note that the average brightness temperature will be less reliable if USE_FCOLL_IONISATION_TABLE is set as it uses only a sub-volume (i.e. larger sample variance)
    if(STORE_DATA) {
        aveTb[sample_index] = ave;
    }

    /////////////////////////////  PRINT OUT THE POWERSPECTRUM  ///////////////////////////////
    if(USE_LF != 2) { // Do NOT compute PS, when wants constraints using (LF + tau_e).

    if(USE_LIGHTCONE) {
            
        if(sample_index==0) {
                
            // NOTE: This only works under the assumption that the hightest redshift is >> larger than reionisation
            // i.e assumes that Z_HEAT_MAX >~ 20.
                
            // Set the highest redshift box for creating the light-cone
            memcpy(box_z2,delta_T,sizeof(float)*HII_TOT_NUM_PIXELS);
            z2_LC = redshifts[sample_index];
            t_z2_LC = gettime(z2_LC);
        }
        else {
                
            // LOS_direction is used in two separate locations (iterated), so we store the state for each individucal usage
            LOS_direction = Stored_LOS_direction_state_2;
                
            memcpy(box_z1,delta_T,sizeof(float)*HII_TOT_NUM_PIXELS);
            z1_LC = redshifts[sample_index];
            t_z1_LC = gettime(z1_LC);
                
            // Box will fill up, therefore, finish writing box, write out full box, continue onto the next box
            if(start_index_LC[N_USER_REDSHIFT-sample_index-1] > end_index_LC[N_USER_REDSHIFT-sample_index-1]) {
                    
                for(ii=0;ii<end_index_LC[N_USER_REDSHIFT-sample_index-1];ii++) {
                    total_slice_ct = num_boxes_interp*HII_DIM + ii;

                    z_slice = slice_redshifts[full_index_LC[total_slice_ct]];
                    t_z_slice = gettime(z_slice);
                        
                    for (j=0;j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            fz1 = box_z1[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz2 = box_z2[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz = (fz2 - fz1) / (t_z2_LC - t_z1_LC) * (t_z_slice - t_z1_LC) + fz1; // linearly interpolate in z (time actually)
                            if(num_boxes_interp==total_num_boxes) {
                                box_interpolate_remainder[ii + remainder_LC*(k+HII_DIM*j)] = fz;
                            }
                            else {
                                box_interpolate[HII_R_INDEX(j, k, ii)] = fz;
                            }
                        }
                    }
                }
                    
                if(num_boxes_interp==total_num_boxes) {
                        
                    if(PRINT_LIGHTCONE_21cmBoxes) {
                        sprintf(filename, "delta_T_%f_%f__zstart%09.5f_zend%09.5f_%i_%.0fMpc_lighttravel",INDIVIDUAL_ID,INDIVIDUAL_ID_2, slice_redshifts[full_index_LC[num_boxes_interp*HII_DIM]], redshifts[0], HII_DIM, BOX_LEN);
                        F = fopen(filename, "wb");
                        mod_fwrite(box_interpolate_remainder, sizeof(float)*(remainder_LC*HII_DIM*HII_DIM), 1, F);
                        fclose(F);
                    }
                    ave = 0.0;
                }
                else {
                        
                    if(PRINT_LIGHTCONE_21cmBoxes) {
                        sprintf(filename, "delta_T_%f_%f__zstart%09.5f_zend%09.5f_%i_%.0fMpc_lighttravel",INDIVIDUAL_ID,INDIVIDUAL_ID_2, slice_redshifts[full_index_LC[num_boxes_interp*HII_DIM]], slice_redshifts[full_index_LC[(num_boxes_interp+1)*HII_DIM]], HII_DIM, BOX_LEN);
                        F = fopen(filename, "wb");
                        mod_fwrite(box_interpolate, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F);
                        fclose(F);
                    }
                        
                    // We have filled the light-cone cube box, now generate the 21cm PS.
                    GeneratePS(0,0.0);
                    
                    // now lets print out the k bins
                    if(PRINT_FILES) {
                            
                        sprintf(filename, "delTps_estimate_%f_%f_zstart%09.5f_zend%09.5f_%i_%.0fMpc_lighttravel.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2,slice_redshifts[full_index_LC[num_boxes_interp*HII_DIM]], slice_redshifts[full_index_LC[(num_boxes_interp+1)*HII_DIM]], HII_DIM, BOX_LEN);
                        F=fopen(filename, "wt");
                        for (ct=1; ct<NUM_BINS; ct++){
                            if (in_bin_ct[ct]>0)
                                fprintf(F, "%e\t%e\t%e\n", k_ave[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0)/sqrt(in_bin_ct[ct]+0.0));
                        }
                        fclose(F);
                            
                        strcpy(lightcone_box_names[num_boxes_interp], filename);
                    }
                }
                    
                num_boxes_interp--;
                    
                // Storing the LOS direction for creating the light-cones (reduced calculation flag uses the same code, so double counts changes to LOS_direction. Hence, we store the state
                Stored_LOS_direction_state_2 = LOS_direction;
                    
                for(ii=start_index_LC[N_USER_REDSHIFT-sample_index-1];ii<HII_DIM;ii++) {
                    // (num_boxes_interp-1) as we are keeping track of the boxes in descending order (num_boxes_interp is decreased by one, further down)
                    // Yes, I know this is the counter intuitive way of doing it, but it renders Andrei's original code mostly intact
                    total_slice_ct = num_boxes_interp*HII_DIM + ii;
                        
                    z_slice = slice_redshifts[full_index_LC[total_slice_ct]];
                    t_z_slice = gettime(z_slice);

                    for (j=0;j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            fz1 = box_z1[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz2 = box_z2[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz = (fz2 - fz1) / (t_z2_LC - t_z1_LC) * (t_z_slice - t_z1_LC) + fz1; // linearly interpolate in z (time actually)
                            if(num_boxes_interp==total_num_boxes) {
                                box_interpolate_remainder[ii + remainder_LC*(k+HII_DIM*j)] = fz;
                            }
                            else {
                                box_interpolate[HII_R_INDEX(j, k, ii)] = fz;
                            }
                        }
                    }
                }
            }
            else{
                    
                for(ii=start_index_LC[N_USER_REDSHIFT-sample_index-1];ii<end_index_LC[N_USER_REDSHIFT-sample_index-1];ii++) {
                    total_slice_ct = num_boxes_interp*HII_DIM + ii;
                        
                    z_slice = slice_redshifts[full_index_LC[total_slice_ct]];
                    t_z_slice = gettime(z_slice);

                    for (j=0;j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            fz1 = box_z1[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz2 = box_z2[coeval_box_pos(LOS_direction,j,k,ii)];
                            fz = (fz2 - fz1) / (t_z2_LC - t_z1_LC) * (t_z_slice - t_z1_LC) + fz1; // linearly interpolate in z (time actually)
                            if(num_boxes_interp==total_num_boxes) {
                                box_interpolate_remainder[ii + remainder_LC*(k+HII_DIM*j)] = fz;
                            }
                            else {
                                box_interpolate[HII_R_INDEX(j, k, ii)] = fz;
                            }
                        }
                    }
                }
            }
            // ok, we are done with this box, move onto the next box
            memcpy(box_z2,delta_T,sizeof(float)*HII_TOT_NUM_PIXELS);
            z2_LC = z1_LC;
            t_z2_LC = gettime(z2_LC);
                
            if((num_boxes_interp==0)&&(sample_index==(N_USER_REDSHIFT-1))) {
                    
                if(PRINT_LIGHTCONE_21cmBoxes) {
                    sprintf(filename, "delta_T_%f_%f__zstart%09.5f_zend%09.5f_%i_%.0fMpc_lighttravel",INDIVIDUAL_ID,INDIVIDUAL_ID_2, slice_redshifts[full_index_LC[num_boxes_interp*HII_DIM]], slice_redshifts[full_index_LC[(num_boxes_interp+1)*HII_DIM]], HII_DIM, BOX_LEN);
                    F = fopen(filename, "wb");
                    mod_fwrite(box_interpolate, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F);
                    fclose(F);
                }
                    
                // We have filled the light-cone cube box, now generate the 21cm PS.
                GeneratePS(0,0.0);
                
                // now lets print out the k bins
                if(PRINT_FILES) {
                    sprintf(filename, "delTps_estimate_%f_%f_zstart%09.5f_zend%09.5f_%i_%.0fMpc_lighttravel.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2,slice_redshifts[full_index_LC[num_boxes_interp*HII_DIM]], slice_redshifts[full_index_LC[(num_boxes_interp+1)*HII_DIM]], HII_DIM, BOX_LEN);
                    F=fopen(filename, "wt");
                    for (ct=1; ct<NUM_BINS; ct++){
                        if (in_bin_ct[ct]>0)
                            fprintf(F, "%e\t%e\t%e\n", k_ave[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0)/sqrt(in_bin_ct[ct]+0.0));
                    }
                    fclose(F);
                        
                    strcpy(lightcone_box_names[num_boxes_interp], filename);
                }
            }
        }
    }

    if(!USE_LIGHTCONE) {

        if(PRINT_COEVAL_21cmBoxes) {
            sprintf(filename, "delta_T_%f_%f_z%1.6_%i_%.0fMpc",INDIVIDUAL_ID,INDIVIDUAL_ID_2, REDSHIFT_SAMPLE, HII_DIM, BOX_LEN);
            F = fopen(filename, "wb");
            mod_fwrite(delta_T, sizeof(float)*HII_TOT_NUM_PIXELS, 1, F);
            fclose(F);
        }

        GeneratePS(1,ave);

        // now lets print out the k bins
        if(PRINT_FILES) {
            sprintf(filename, "delTps_estimate_%f_%f_%f.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2,REDSHIFT_SAMPLE);
            F=fopen(filename, "wt");
            for (ct=1; ct<NUM_BINS; ct++){
                if (in_bin_ct[ct]>0)
                    fprintf(F, "%e\t%e\t%e\n", k_ave[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0), p_box[ct]/(in_bin_ct[ct]+0.0)/sqrt(in_bin_ct[ct]+0.0));
            }
            fclose(F);
        }
    ///////////////////////////// END POWER SPECTRUM STUFF   ////////////////////////////////////

    //////////////////////////// End of perform 'delta_T.c' /////////////////////////////////////
    }
    } // Do NOT compute PS, when wants constraints using (LF + tau_e).
    
    free(LOS_index);
    free(slice_index);
    
    destroy_21cmMC_HII_arrays(skip_deallocate);
    return nf;    
}

void ComputeInitialConditions() {
    
    /*
     Generates the initial conditions: gaussian random density field (DIM^3) as well as the equal or lower resolution velocity fields, and smoothed density field (HII_DIM^3).
     See INIT_PARAMS.H and ANAL_PARAMS.H to set the appropriate parameters.
     Output is written to ../Boxes
     
     Author: Andrei Mesinger
     Date: 9/29/06
     */
    
    fftwf_plan plan;
    
    char wisdom_filename[500];
    
    unsigned long long ct;
    int n_x, n_y, n_z, i, j, k, ii;
    float k_x, k_y, k_z, k_mag, p, a, b, k_sq;
    double pixel_deltax;
    
    float f_pixel_factor;
    
    gsl_rng * r;

    /************  INITIALIZATION **********************/
    
    // Removed all references to threads as 21CMMC is always a single core implementation
    
    // seed the random number generators
    r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, RANDOM_SEED);
        
    // allocate array for the k-space and real-space boxes
    HIRES_box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    HIRES_box_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    
    // now allocate memory for the lower-resolution box
    // use HII_DIM from ANAL_PARAMS
    LOWRES_density = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    LOWRES_vx = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    LOWRES_vy= (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    LOWRES_vz = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    
    if(SECOND_ORDER_LPT_CORRECTIONS){
        LOWRES_vx_2LPT = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        LOWRES_vy_2LPT = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
        LOWRES_vz_2LPT = (float *) malloc(sizeof(float)*HII_TOT_NUM_PIXELS);
    }
    
    // find factor of HII pixel size / deltax pixel size
    f_pixel_factor = DIM/(float)HII_DIM;
    /************  END INITIALIZATION ******************/
    
    /************ CREATE K-SPACE GAUSSIAN RANDOM FIELD ***********/
    for (n_x=0; n_x<DIM; n_x++){
        // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
        if (n_x>MIDDLE)
            k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
        else
            k_x = n_x * DELTA_K;
                
        for (n_y=0; n_y<DIM; n_y++){
            // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
            if (n_y>MIDDLE)
                k_y =(n_y-DIM) * DELTA_K;
            else
                k_y = n_y * DELTA_K;
                    
            // since physical space field is real, only half contains independent modes
            for (n_z=0; n_z<=MIDDLE; n_z++){
                // convert index to numerical value for this component of the k-mode: k = (2*pi/L) * n
                k_z = n_z * DELTA_K;
                        
                // now get the power spectrum; remember, only the magnitude of k counts (due to issotropy)
                // this could be used to speed-up later maybe
                k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
                p = power_in_k(k_mag);
                        
                // ok, now we can draw the values of the real and imaginary part
                // of our k entry from a Gaussian distribution
                a = gsl_ran_ugaussian(r);
                b = gsl_ran_ugaussian(r);
                HIRES_box[C_INDEX(n_x, n_y, n_z)] = sqrt(VOLUME*p/2.0) * (a + b*I);
            }
        }
    }
        
    /*****  Adjust the complex conjugate relations for a real array  *****/
    adj_complex_conj();
        
    /*** Let's also create a lower-resolution version of the density field  ***/
    
    memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    
    if (DIM != HII_DIM)
        filter(HIRES_box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));
    
    // FFT back to real space
    if(USE_FFTW_WISDOM) {
        // Check to see if wisdom exists, if not create it
        sprintf(wisdom_filename,"../FFTW_Wisdoms/complex_to_real_%d.fftwf_wisdom",DIM);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
            plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_PATIENT);
            fftwf_execute(plan);
            
            // Store the wisdom for later use
            fftwf_export_wisdom_to_filename(wisdom_filename);
            
            memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            
            plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
    
    // now sample the filtered box
    for (i=0; i<HII_DIM; i++){
        for (j=0; j<HII_DIM; j++){
            for (k=0; k<HII_DIM; k++){
                LOWRES_density[HII_R_INDEX(i,j,k)] =
                *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                             (unsigned long long)(j*f_pixel_factor+0.5),
                                             (unsigned long long)(k*f_pixel_factor+0.5)))/VOLUME;
            }
        }
    }
    /******* PERFORM INVERSE FOURIER TRANSFORM *****************/
    // add the 1/VOLUME factor when converting from k space to real space

    memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
    
    for (ct=0; ct<KSPACE_NUM_PIXELS; ct++){
        HIRES_box[ct] /= VOLUME;
    }
    
    if(USE_FFTW_WISDOM) {
        // Wisdom should have been created earlier if needed
        plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
        fftwf_execute(plan);
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
//    fftwf_destroy_plan(plan);
    
    for (i=0; i<DIM; i++){
        for (j=0; j<DIM; j++){
            for (k=0; k<DIM; k++){
                *((float *)HIRES_density + R_FFT_INDEX(i,j,k)) = *((float *)HIRES_box + R_FFT_INDEX(i,j,k));
            }
        }
    }
    
    for(ii=0;ii<3;ii++) {
    
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        /*** Now let's set the velocity field/dD/dt (in comoving Mpc) ***/
        
        for (n_x=0; n_x<DIM; n_x++){
            if (n_x>MIDDLE)
                k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;
                
            for (n_y=0; n_y<DIM; n_y++){
                if (n_y>MIDDLE)
                    k_y =(n_y-DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;
                    
                for (n_z=0; n_z<=MIDDLE; n_z++){
                    k_z = n_z * DELTA_K;
                
                    k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                        
                    // now set the velocities
                    if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                        HIRES_box[0] = 0.;
                    }
                    else{
                        if(ii==0) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq/VOLUME;
                        }
                        if(ii==1) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq/VOLUME;
                        }
                        if(ii==2) {
                            HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq/VOLUME;
                        }
                        // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                    }
                }
            }
        }
        
        if (DIM != HII_DIM)
            filter(HIRES_box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));

        if(USE_FFTW_WISDOM) {
            // Wisdom should have been created earlier if needed
            plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        // now sample to lower res
        // now sample the filtered box
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    if(ii==0) {
                        LOWRES_vx[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                    if(ii==1) {
                        LOWRES_vy[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                    if(ii==2) {
                        LOWRES_vz[HII_R_INDEX(i,j,k)] =
                        *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                           (unsigned long long)(j*f_pixel_factor+0.5),
                                                           (unsigned long long)(k*f_pixel_factor+0.5)));
                    }
                }
            }
        }
    }
    // write out file
        
    /* *************************************************** *
     *              BEGIN 2LPT PART                        *
     * *************************************************** */
        
    // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to the ZA
    // reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        
    // Parameter set in ANAL_PARAMS.H
    if(SECOND_ORDER_LPT_CORRECTIONS){
        // use six supplementary boxes to store the gradients of phi_1 (eq. D13b)
        // Allocating the boxes
#define PHI_INDEX(i, j) ((int) ((i) - (j)) + 3*((j)) - ((int)(j))/2  )
        // ij -> INDEX
        // 00 -> 0
        // 11 -> 3
        // 22 -> 5
        // 10 -> 1
        // 20 -> 2
        // 21 -> 4
            
        fftwf_complex *phi_1[6];
            
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                phi_1[PHI_INDEX(i, j)] = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            }
        }
            
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                    
                // read in the box
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
                    
                // generate the phi_1 boxes in Fourier transform
                for (n_x=0; n_x<DIM; n_x++){
                    if (n_x>MIDDLE)
                        k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
                    else
                        k_x = n_x * DELTA_K;
                            
                    for (n_y=0; n_y<DIM; n_y++){
                        if (n_y>MIDDLE)
                            k_y =(n_y-DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;
                            
                        for (n_z=0; n_z<=MIDDLE; n_z++){
                            k_z = n_z * DELTA_K;
                                    
                            k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                                    
                            float k[] = {k_x, k_y, k_z};
                            // now set the velocities
                            if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                                phi_1[PHI_INDEX(i, j)][0] = 0;
                            }
                            else{
                                phi_1[PHI_INDEX(i, j)][C_INDEX(n_x,n_y,n_z)] = -k[i]*k[j]*HIRES_box[C_INDEX(n_x, n_y, n_z)]/k_sq/VOLUME;
                                // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                            }
                        }
                    }
                }
                // Now we can generate the real phi_1[i,j]
                if(USE_FFTW_WISDOM) {
                    // Wisdom should have been created earlier if needed
                    plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)phi_1[PHI_INDEX(i, j)], (float *)phi_1[PHI_INDEX(i, j)], FFTW_WISDOM_ONLY);
                    fftwf_execute(plan);
                }
                else {
                    plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)phi_1[PHI_INDEX(i, j)], (float *)phi_1[PHI_INDEX(i, j)], FFTW_ESTIMATE);
                    fftwf_execute(plan);
                }
            }
        }
            
        // Then we will have the laplacian of phi_2 (eq. D13b)
        // After that we have to return in Fourier space and generate the Fourier transform of phi_2
        int m, l;
        for (i=0; i<DIM; i++){
            for (j=0; j<DIM; j++){
                for (k=0; k<DIM; k++){
                    *( (float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i), (unsigned long long)(j), (unsigned long long)(k) )) = 0.0;
                    for(m = 0; m < 3; ++m){
                        for(l = m+1; l < 3; ++l){
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) += ( *((float *)(phi_1[PHI_INDEX(l, l)]) + R_FFT_INDEX((unsigned long long) (i),(unsigned long long) (j),(unsigned long long) (k)))  ) * (  *((float *)(phi_1[PHI_INDEX(m, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)))  );
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) -= ( *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long) (j),(unsigned long long)(k) ) )  ) * (  *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k) ))  );
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;
                        }
                    }
                }
            }
        }
            
        // Perform FFTs
        if(USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"../FFTW_Wisdoms/read_to_complex_%d.fftwf_wisdom",DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(DIM, DIM, DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(DIM, DIM, DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
                
                // Repeating the above computation as the creating the wisdom overwrites the input data
                
                // Then we will have the laplacian of phi_2 (eq. D13b)
                // After that we have to return in Fourier space and generate the Fourier transform of phi_2
                int m, l;
                for (i=0; i<DIM; i++){
                    for (j=0; j<DIM; j++){
                        for (k=0; k<DIM; k++){
                            *( (float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i), (unsigned long long)(j), (unsigned long long)(k) )) = 0.0;
                            for(m = 0; m < 3; ++m){
                                for(l = m+1; l < 3; ++l){
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) += ( *((float *)(phi_1[PHI_INDEX(l, l)]) + R_FFT_INDEX((unsigned long long) (i),(unsigned long long) (j),(unsigned long long) (k)))  ) * (  *((float *)(phi_1[PHI_INDEX(m, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)))  );
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) -= ( *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long) (j),(unsigned long long)(k) ) )  ) * (  *((float *)(phi_1[PHI_INDEX(l, m)]) + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k) ))  );
                                    *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i),(unsigned long long)(j),(unsigned long long)(k)) ) /= TOT_NUM_PIXELS;
                                }
                            }
                        }
                    }
                }
                
                plan = fftwf_plan_dft_r2c_3d(DIM, DIM, DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(DIM, DIM, DIM, (float *)HIRES_box, (fftwf_complex *)HIRES_box, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
        
        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
        
        // Now we can store the content of box in a back-up file
        // Then we can generate the gradients of phi_2 (eq. D13b and D9)
            
        /***** Write out back-up k-box RHS eq. D13b *****/
            
        // For each component, we generate the velocity field (same as the ZA part)
            
        /*** Now let's set the velocity field/dD/dt (in comoving Mpc) ***/
            
        // read in the box
        // TODO correct free of phi_1

        for(ii=0;ii<3;ii++) {

            if(ii>0) {
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex)*KSPACE_NUM_PIXELS);
            }
            // set velocities/dD/dt
            for (n_x=0; n_x<DIM; n_x++){
                if (n_x>MIDDLE)
                    k_x =(n_x-DIM) * DELTA_K;  // wrap around for FFT convention
                else
                    k_x = n_x * DELTA_K;
                
                for (n_y=0; n_y<DIM; n_y++){
                    if (n_y>MIDDLE)
                        k_y =(n_y-DIM) * DELTA_K;
                    else
                        k_y = n_y * DELTA_K;
                        
                    for (n_z=0; n_z<=MIDDLE; n_z++){
                        k_z = n_z * DELTA_K;
                            
                        k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                            
                        // now set the velocities
                        if ((n_x==0) && (n_y==0) && (n_z==0)){ // DC mode
                            HIRES_box[0] = 0.;
                        }
                        else{
                            if(ii==0) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_x*I/k_sq;
                            }
                            if(ii==1) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_y*I/k_sq;
                            }
                            if(ii==2) {
                                HIRES_box[C_INDEX(n_x,n_y,n_z)] *= k_z*I/k_sq;
                            }
                            // note the last factor of 1/VOLUME accounts for the scaling in real-space, following the FFT
                        }
                    }
                }
            }
            
            if (DIM != HII_DIM)
                filter(HIRES_box, 0, L_FACTOR*BOX_LEN/(HII_DIM+0.0));

            if(USE_FFTW_WISDOM) {
                // Wisdom should have been created earlier if needed
                plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_c2r_3d(DIM, DIM, DIM, (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_ESTIMATE);
                fftwf_execute(plan);
            }
            // now sample to lower res
            // now sample the filtered box
            for (i=0; i<HII_DIM; i++){
                for (j=0; j<HII_DIM; j++){
                    for (k=0; k<HII_DIM; k++){
                        if(ii==0) {
                            LOWRES_vx_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                        if(ii==1) {
                            LOWRES_vy_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                        if(ii==2) {
                            LOWRES_vz_2LPT[HII_R_INDEX(i,j,k)] =
                            *((float *)HIRES_box + R_FFT_INDEX((unsigned long long)(i*f_pixel_factor+0.5),
                                                               (unsigned long long)(j*f_pixel_factor+0.5),
                                                               (unsigned long long)(k*f_pixel_factor+0.5)));
                        }
                    }
                }
            }
        }
        // deallocate the supplementary boxes
        for(i = 0; i < 3; ++i){
            for(j = 0; j <= i; ++j){
                fftwf_free(phi_1[PHI_INDEX(i,j)]);
            }
        }
    }
    /* *********************************************** *
     *               END 2LPT PART                     *
     * *********************************************** */
    
    // deallocate
    fftwf_free(HIRES_box);
    fftwf_free(HIRES_box_saved);
}

/*****  Adjust the complex conjugate relations for a real array  *****/
void adj_complex_conj(){
    int i, j, k;
    
    // corners
    HIRES_box[C_INDEX(0,0,0)] = 0;
    HIRES_box[C_INDEX(0,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,0,MIDDLE)]);
    HIRES_box[C_INDEX(0,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,0)]);
    HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(0,MIDDLE,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,0,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,0)]);
    HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,0,MIDDLE)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,0)]);
    HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)] = crealf(HIRES_box[C_INDEX(MIDDLE,MIDDLE,MIDDLE)]);
    
    // do entire i except corners
    for (i=1; i<MIDDLE; i++){
        // just j corners
        for (j=0; j<=MIDDLE; j+=MIDDLE){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX(DIM-i,j,k)]);
            }
        }
        
        // all of j
        for (j=1; j<MIDDLE; j++){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX(DIM-i,DIM-j,k)]);
                HIRES_box[C_INDEX(i,DIM-j,k)] = conjf(HIRES_box[C_INDEX(DIM-i,j,k)]);
            }
        }
    } // end loop over i
    
    // now the i corners
    for (i=0; i<=MIDDLE; i+=MIDDLE){
        for (j=1; j<MIDDLE; j++){
            for (k=0; k<=MIDDLE; k+=MIDDLE){
                HIRES_box[C_INDEX(i,j,k)] = conjf(HIRES_box[C_INDEX(i,DIM-j,k)]);
            }
        }
    } // end loop over remaining j
}

void ComputePerturbField(float REDSHIFT_SAMPLE) {
    
    /*
     USAGE: perturb_field <REDSHIFT>
     
     PROGRAM PERTURB_FIELD uses the first-order Langragian displacement field to move the masses in the cells of the density field.
     The high-res density field is extrapolated to some high-redshift (INITIAL_REDSHIFT in ANAL_PARAMS.H), then uses the zeldovich approximation
     to move the grid "particles" onto the lower-res grid we use for the maps.  Then we recalculate the velocity fields on the perturbed grid.
     */
    
    fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;
    fftwf_plan plan;
    
    char wisdom_filename[500];
    
    float REDSHIFT, growth_factor, displacement_factor_2LPT, init_growth_factor, init_displacement_factor_2LPT, xf, yf, zf;
    float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor, velocity_displacement_factor_2LPT;
    unsigned long long ct, HII_i, HII_j, HII_k;
    int i,j,k, xi, yi, zi;
    double ave_delta, new_ave_delta;
    /***************   BEGIN INITIALIZATION   **************************/
    
    // perform a very rudimentary check to see if we are underresolved and not using the linear approx
    if ((BOX_LEN > DIM) && !EVOLVE_DENSITY_LINEARLY){
        fprintf(stderr, "perturb_field.c: WARNING: Resolution is likely too low for accurate evolved density fields\n It Is recommended that you either increase the resolution (DIM/Box_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to 1\n");
    }
        
    growth_factor = dicke(REDSHIFT_SAMPLE);
    displacement_factor_2LPT = -(3.0/7.0) * growth_factor*growth_factor; // 2LPT eq. D8
        
    dDdt = ddickedt(REDSHIFT_SAMPLE); // time derivative of the growth factor (1/s)
    init_growth_factor = dicke(INITIAL_REDSHIFT);
    init_displacement_factor_2LPT = -(3.0/7.0) * init_growth_factor*init_growth_factor; // 2LPT eq. D8
        
    // allocate memory for the updated density, and initialize
    LOWRES_density_perturb = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    LOWRES_density_perturb_saved = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    // check if the linear evolution flag was set
    if (EVOLVE_DENSITY_LINEARLY){
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = growth_factor*LOWRES_density[HII_R_INDEX(i,j,k)];
                }
            }
        }
    }
    // first order Zel'Dovich perturbation
    else{
        
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = 0.;
                }
            }
        }
        
        velocity_displacement_factor = (growth_factor-init_growth_factor) / BOX_LEN;
        
        // now add the missing factor of D
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            LOWRES_vx[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
            LOWRES_vy[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
            LOWRES_vz[ct] *= velocity_displacement_factor; // this is now comoving displacement in units of box size
        }
        
        // find factor of HII pixel size / deltax pixel size
        f_pixel_factor = DIM/(float)HII_DIM;
        mass_factor = pow(f_pixel_factor, 3);
        
        /* ************************************************************************* *
         *                           BEGIN 2LPT PART                                 *
         * ************************************************************************* */
        // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if(SECOND_ORDER_LPT_CORRECTIONS){
                
            // allocate memory for the velocity boxes and read them in
            velocity_displacement_factor_2LPT = (displacement_factor_2LPT - init_displacement_factor_2LPT) / BOX_LEN;
            
            // now add the missing factor in eq. D9
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                LOWRES_vx_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                LOWRES_vy_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
                LOWRES_vz_2LPT[ct] *= velocity_displacement_factor_2LPT; // this is now comoving displacement in units of box size
            }
        }
            
        /* ************************************************************************* *
         *                            END 2LPT PART                                  *
         * ************************************************************************* */
            
        /************  END INITIALIZATION ****************************/
        
        // go through the high-res box, mapping the mass onto the low-res (updated) box
        for (i=0; i<DIM;i++){
            for (j=0; j<DIM;j++){
                for (k=0; k<DIM;k++){
                        
                    // map indeces to locations in units of box size
                    xf = (i+0.5)/(DIM+0.0);
                    yf = (j+0.5)/(DIM+0.0);
                    zf = (k+0.5)/(DIM+0.0);
                        
                    // update locations
                    HII_i = (unsigned long long)(i/f_pixel_factor);
                    HII_j = (unsigned long long)(j/f_pixel_factor);
                    HII_k = (unsigned long long)(k/f_pixel_factor);
                    xf += LOWRES_vx[HII_R_INDEX(HII_i, HII_j, HII_k)];
                    yf += LOWRES_vy[HII_R_INDEX(HII_i, HII_j, HII_k)];
                    zf += LOWRES_vz[HII_R_INDEX(HII_i, HII_j, HII_k)];
                    
                    // 2LPT PART
                    // add second order corrections
                    if(SECOND_ORDER_LPT_CORRECTIONS){
                        xf -= LOWRES_vx_2LPT[HII_R_INDEX(HII_i,HII_j,HII_k)];
                        yf -= LOWRES_vy_2LPT[HII_R_INDEX(HII_i,HII_j,HII_k)];
                        zf -= LOWRES_vz_2LPT[HII_R_INDEX(HII_i,HII_j,HII_k)];
                    }
                        
                    xf *= HII_DIM;
                    yf *= HII_DIM;
                    zf *= HII_DIM;
                    while (xf >= (float)HII_DIM){ xf -= HII_DIM;}
                    while (xf < 0){ xf += HII_DIM;}
                    while (yf >= (float)HII_DIM){ yf -= HII_DIM;}
                    while (yf < 0){ yf += HII_DIM;}
                    while (zf >= (float)HII_DIM){ zf -= HII_DIM;}
                    while (zf < 0){ zf += HII_DIM;}
                    xi = xf;
                    yi = yf;
                    zi = zf;
                    if (xi >= HII_DIM){ xi -= HII_DIM;}
                    if (xi < 0) {xi += HII_DIM;}
                    if (yi >= HII_DIM){ yi -= HII_DIM;}
                    if (yi < 0) {yi += HII_DIM;}
                    if (zi >= HII_DIM){ zi -= HII_DIM;}
                    if (zi < 0) {zi += HII_DIM;}
                        
                    // now move the mass
                    *( (float *)LOWRES_density_perturb + HII_R_FFT_INDEX(xi, yi, zi) ) +=
                    (1 + init_growth_factor*HIRES_density[R_FFT_INDEX(i,j,k)]);
                }
            }
        }
            
        // renormalize to the new pixel size, and make into delta
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) /= mass_factor;
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k) ) -= 1;
                }
            }
        }
        
        // deallocate
        for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
            LOWRES_vx[ct] /= velocity_displacement_factor; // convert back to z = 0 quantity
            LOWRES_vy[ct] /= velocity_displacement_factor;
            LOWRES_vz[ct] /= velocity_displacement_factor;
        }
        
        if(SECOND_ORDER_LPT_CORRECTIONS){
            for (ct=0; ct<HII_TOT_NUM_PIXELS; ct++){
                LOWRES_vx_2LPT[ct] /= velocity_displacement_factor_2LPT; // convert back to z = 0 quantity
                LOWRES_vy_2LPT[ct] /= velocity_displacement_factor_2LPT;
                LOWRES_vz_2LPT[ct] /= velocity_displacement_factor_2LPT;
            }
        }
    }
    
    /****  Print and convert to velocities *****/
    if (EVOLVE_DENSITY_LINEARLY){
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_REDSHIFT + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
        
        // transform to k-space
        if(USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"../FFTW_Wisdoms/real_to_complex_%d.fftwf_wisdom",HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                // Now need to re-fill the LOWRES_density_perturb cube as it was overwritten creating the FFTW wisdom
                for (i=0; i<HII_DIM; i++){
                    for (j=0; j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = *((float *)LOWRES_density_REDSHIFT + HII_R_INDEX(i,j,k));
                        }
                    }
                }
                
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
//        fftwf_destroy_plan(plan);
        
        // save a copy of the k-space density field
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    }
    else{
        // transform to k-space
        if(USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"../FFTW_Wisdoms/real_to_complex_%d.fftwf_wisdom",HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                
                // Temporarily save the data prior to creating the wisdom (as it'll be overwritten). The LOWRES_density_REDSHIFT cube will be properly filled lower down
                for (i=0; i<HII_DIM; i++){
                    for (j=0; j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            *((float *)LOWRES_density_REDSHIFT + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                        }
                    }
                }
                
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                // Now need to re-fill the LOWRES_density_perturb cube as it was overwritten creating the FFTW wisdom
                for (i=0; i<HII_DIM; i++){
                    for (j=0; j<HII_DIM; j++){
                        for (k=0; k<HII_DIM; k++){
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = *((float *)LOWRES_density_REDSHIFT + HII_R_INDEX(i,j,k));
                        }
                    }
                }
                
                plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)LOWRES_density_perturb, (fftwf_complex *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
//        fftwf_destroy_plan(plan);
            
        //smooth the field
        if (!EVOLVE_DENSITY_LINEARLY && SMOOTH_EVOLVED_DENSITY_FIELD){
            HII_filter(LOWRES_density_perturb, 2, R_smooth_density*BOX_LEN/(float)HII_DIM);
        }
            
        // save a copy of the k-space density field
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        
        if(USE_FFTW_WISDOM) {
            // Check to see if wisdom exists, if not create it
            sprintf(wisdom_filename,"../FFTW_Wisdoms/complex_to_real_%d.fftwf_wisdom",HII_DIM);
            if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
            else {
                
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_PATIENT);
                fftwf_execute(plan);
                
                // Store the wisdom for later use
                fftwf_export_wisdom_to_filename(wisdom_filename);
                
                memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
                
                plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
                fftwf_execute(plan);
            }
        }
        else {
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
            fftwf_execute(plan);
        }
//        fftwf_destroy_plan(plan);
            
        // normalize after FFT
        for(i=0; i<HII_DIM; i++){
            for(j=0; j<HII_DIM; j++){
                for(k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) /= (float)HII_TOT_NUM_PIXELS;
                    if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) < -1) // shouldn't happen
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k)) = -1.+FRACT_FLOAT_ERR;
                }
            }
        }
        
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    *((float *)LOWRES_density_REDSHIFT + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
                }
            }
        }
        memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    }
    
    float k_x, k_y, k_z, k_sq, dDdt_over_D;
    int n_x, n_y, n_z;
    
    dDdt_over_D = dDdt/growth_factor;
    
    for (n_x=0; n_x<HII_DIM; n_x++){
        if (n_x>HII_MIDDLE)
            k_x =(n_x-HII_DIM) * DELTA_K;  // wrap around for FFT convention
        else
            k_x = n_x * DELTA_K;
        
        for (n_y=0; n_y<HII_DIM; n_y++){
            if (n_y>HII_MIDDLE)
                k_y =(n_y-HII_DIM) * DELTA_K;
            else
                k_y = n_y * DELTA_K;
            
            for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                k_z = n_z * DELTA_K;
                
                k_sq = k_x*k_x + k_y*k_y + k_z*k_z;
                
                // now set the velocities
                if ((n_x==0) && (n_y==0) && (n_z==0)) // DC mode
                    LOWRES_density_perturb[0] = 0;
                else{
                    if(VELOCITY_COMPONENT==1) // x-component
                        LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_x*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                    else if (VELOCITY_COMPONENT == 2)
                        LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_y*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
                    else
                        LOWRES_density_perturb[HII_C_INDEX(n_x,n_y,n_z)] *= dDdt_over_D*k_z*I/k_sq/(HII_TOT_NUM_PIXELS+0.0);
               }
            }
        }
    }
    
    if(USE_FFTW_WISDOM) {
        // Check to see if wisdom exists, if not create it
        sprintf(wisdom_filename,"../FFTW_Wisdoms/complex_to_real_%d.fftwf_wisdom",HII_DIM);
        if(fftwf_import_wisdom_from_filename(wisdom_filename)!=0) {
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
        else {
            
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_PATIENT);
            fftwf_execute(plan);
            
            // Store the wisdom for later use
            fftwf_export_wisdom_to_filename(wisdom_filename);
            
            memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved, sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
            
            plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_WISDOM_ONLY);
            fftwf_execute(plan);
        }
    }
    else {
        plan = fftwf_plan_dft_c2r_3d(HII_DIM, HII_DIM, HII_DIM, (fftwf_complex *)LOWRES_density_perturb, (float *)LOWRES_density_perturb, FFTW_ESTIMATE);
        fftwf_execute(plan);
    }
//    fftwf_destroy_plan(plan);
    
    for (i=0; i<HII_DIM; i++){
        for (j=0; j<HII_DIM; j++){
            for (k=0; k<HII_DIM; k++){
                *((float *)LOWRES_velocity_REDSHIFT + HII_R_INDEX(i,j,k)) = *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i,j,k));
            }
        }
    }
    
    // deallocate
    fftwf_free(LOWRES_density_perturb);
    fftwf_free(LOWRES_density_perturb_saved);
}


void ReadFcollTable() {
    
    char filename[500];
    char dummy_string[500];
    FILE *F;
    
    double *PARAM_VALS_FCOLL = (double*) calloc(TOTAL_AVAILABLE_PARAMS_FCOLL_TABLE,sizeof(double));
    
    int i;
    
    // Note the below code does not yet work for ALPHA != 0
    
    sprintf(filename,"f_coll_lightcone_data_%d_%.0fMpc.txt", HII_DIM, BOX_LEN);
    F = fopen(filename,"rt");
    
    for(i=0;i<TOTAL_AVAILABLE_PARAMS_FCOLL_TABLE;i++) {
        fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_VALS_FCOLL[i]);
    }
    fclose(F);
    
    R_MFP_UB = PARAM_VALS_FCOLL[0];
    TVIR_LB_FCOLL = PARAM_VALS_FCOLL[1];
    TVIR_UB_FCOLL = PARAM_VALS_FCOLL[2];
    ZETA_PL_LB = PARAM_VALS_FCOLL[3];
    ZETA_PL_UB = PARAM_VALS_FCOLL[4];
    R_MFP_STEPS = (int)PARAM_VALS_FCOLL[5];
    TVIR_STEPS = (int)PARAM_VALS_FCOLL[6];
    PL_STEPS = (int)PARAM_VALS_FCOLL[7];
    
    R_MFP_MIN = fmax(R_BUBBLE_MIN, (L_FACTOR*BOX_LEN/(float)HII_DIM));
    
    R_MFP_BINWIDTH = ( R_MFP_UB - R_MFP_MIN )/((float)R_MFP_STEPS - 1.);
    TVIR_BINWIDTH = ( TVIR_UB_FCOLL - TVIR_LB_FCOLL )/((float)TVIR_STEPS - 1.);
    
    PL_BINWIDTH = ( ZETA_PL_UB - ZETA_PL_LB )/((float)PL_STEPS - 1.);
    
    TVIR_INT_1 = (int)floor((log10(ION_Tvir_MIN) - TVIR_LB_FCOLL)/TVIR_BINWIDTH);
    TVIR_INT_2 = TVIR_INT_1 + 1;
    
    TVIR_VAL_1 = TVIR_LB_FCOLL + ( TVIR_UB_FCOLL - TVIR_LB_FCOLL )*(float)TVIR_INT_1/((float)TVIR_STEPS - 1.);
    TVIR_VAL_2 = TVIR_LB_FCOLL + ( TVIR_UB_FCOLL - TVIR_LB_FCOLL )*(float)TVIR_INT_2/((float)TVIR_STEPS - 1.);
    
    ZETA_PL_INT_1 = (int)floor((EFF_FACTOR_PL_INDEX - ZETA_PL_LB)/PL_BINWIDTH);
    ZETA_PL_INT_2 = ZETA_PL_INT_1 + 1;
    
    ZETA_PL_VAL_1 = ZETA_PL_LB + ( ZETA_PL_UB - ZETA_PL_LB )*(float)ZETA_PL_INT_1/((float)PL_STEPS - 1.);
    ZETA_PL_VAL_2 = ZETA_PL_LB + ( ZETA_PL_UB - ZETA_PL_LB )*(float)ZETA_PL_INT_2/((float)PL_STEPS - 1.);
    
    if(INCLUDE_ZETA_PL) {
        SIZE_FIRST = N_USER_REDSHIFT*R_MFP_STEPS*TVIR_STEPS*PL_STEPS;
        SIZE_FINAL = N_USER_REDSHIFT*TVIR_STEPS*PL_STEPS;
    }
    else {
        SIZE_FIRST = N_USER_REDSHIFT*R_MFP_STEPS*TVIR_STEPS;
        SIZE_FINAL = N_USER_REDSHIFT*TVIR_STEPS;
    }
    
    Ionisation_fcoll_table = (double*) calloc(SIZE_FIRST,sizeof(double));
    Ionisation_fcoll_table_final = (double*) calloc(SIZE_FINAL,sizeof(double));
    
    // NOTE: No support for the power law index for zeta at the present time
    
    if(INCLUDE_ZETA_PL) {
        sprintf(filename, "Ionisation_fcoll_table_final_Rmax%f_Tmin%f_Tmax%f_PLmin%1.6f_PLmax%1.6f_%d_%.0fMpc", R_MFP_UB, TVIR_LB_FCOLL, TVIR_UB_FCOLL, ZETA_PL_LB, ZETA_PL_UB, HII_DIM, BOX_LEN);
    }
    else {
        sprintf(filename, "Ionisation_fcoll_table_final_Rmax%f_Tmin%f_Tmax%f_PL%1.6f_%d_%.0fMpc", R_MFP_UB, TVIR_LB_FCOLL, TVIR_UB_FCOLL, EFF_FACTOR_PL_INDEX, HII_DIM, BOX_LEN);
    }
    F = fopen(filename, "rb");
    fread(Ionisation_fcoll_table_final,N_USER_REDSHIFT*TVIR_STEPS,sizeof(double),F);
    fclose(F);
    
    if(INCLUDE_ZETA_PL) {
        sprintf(filename, "Ionisation_fcoll_table_Rmax%f_Tmin%f_Tmax%f_PLmin%1.6f_PLmax%1.6f_%d_%.0fMpc", R_MFP_UB, TVIR_LB_FCOLL, TVIR_UB_FCOLL, ZETA_PL_LB, ZETA_PL_UB, HII_DIM, BOX_LEN);
    }
    else {
        sprintf(filename, "Ionisation_fcoll_table_Rmax%f_Tmin%f_Tmax%f_PL%1.6f_%d_%.0fMpc", R_MFP_UB, TVIR_LB_FCOLL, TVIR_UB_FCOLL, EFF_FACTOR_PL_INDEX, HII_DIM, BOX_LEN);
    }
    F = fopen(filename, "rb");
    fread(Ionisation_fcoll_table,N_USER_REDSHIFT*R_MFP_STEPS*TVIR_STEPS,sizeof(double),F);
    fclose(F);    
}

void GeneratePS(int CO_EVAL, double AverageTb) {
    
    fftwf_plan plan;
    
    int i,j,k,n_x, n_y, n_z, skip_zero_mode;;
    float k_x, k_y, k_z, k_mag;
    double ave;
    unsigned long long ct;

    if(!CO_EVAL) {
    
        ave = 0.0;
        for (i=0; i<HII_DIM; i++){
            for (j=0; j<HII_DIM; j++){
                for (k=0; k<HII_DIM; k++){
                    ave += box_interpolate[HII_R_INDEX(i,j,k)];
                }
            }
        }
        ave /= (HII_TOT_NUM_PIXELS+0.0);
    }
    else {
        ave = AverageTb;
    }

    for (ct=0; ct<NUM_BINS; ct++){
        p_box[ct] = k_ave[ct] = 0;
        in_bin_ct[ct] = 0;
    }

    // fill-up the real-space of the deldel box
    for (i=0; i<HII_DIM; i++){
        for (j=0; j<HII_DIM; j++){
            for (k=0; k<HII_DIM; k++){
                if(!CO_EVAL) {
                    *((float *)deldel_T_LC + HII_R_FFT_INDEX(i,j,k)) = (box_interpolate[HII_R_INDEX(i,j,k)]/ave - 1)*VOLUME/(HII_TOT_NUM_PIXELS+0.0);
                }
                else {
                    *((float *)deldel_T_LC + HII_R_FFT_INDEX(i,j,k)) = (delta_T[HII_R_INDEX(i,j,k)]/ave - 1)*VOLUME/(HII_TOT_NUM_PIXELS+0.0);
                }
                if (DIMENSIONAL_T_POWER_SPEC){
                    *((float *)deldel_T_LC + HII_R_FFT_INDEX(i,j,k)) *= ave;
                }
                // Note: we include the V/N factor for the scaling after the fft
            }
        }
    }

    // transform to k-space
    if(USE_FFTW_WISDOM) {
        plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deldel_T_LC, (fftwf_complex *)deldel_T_LC, FFTW_WISDOM_ONLY);
    }
    else {
        plan = fftwf_plan_dft_r2c_3d(HII_DIM, HII_DIM, HII_DIM, (float *)deldel_T_LC, (fftwf_complex *)deldel_T_LC, FFTW_ESTIMATE);
    }
    fftwf_execute(plan);
//    fftwf_destroy_plan(plan);

    // If the light-cone 21cm PS is to be calculated, one should avoid the k(k_x = 0, k_y = 0, k_z) modes (see Datta et al. 2012).
    if(!CO_EVAL) {
     
        // now construct the power spectrum file
        for (n_x=0; n_x<HII_DIM; n_x++){
            if (n_x>HII_MIDDLE)
                k_x =(n_x-HII_DIM) * DELTA_K;  // wrap around for FFT convention
            else 
                k_x = n_x * DELTA_K;
     
            for (n_y=0; n_y<HII_DIM; n_y++){
     
                // avoid the k(k_x = 0, k_y = 0, k_z) modes
                if(n_x != 0 && n_y != 0) { 
     
                    if (n_y>HII_MIDDLE)
                        k_y =(n_y-HII_DIM) * DELTA_K;
                    else 
                        k_y = n_y * DELTA_K;
     
                    for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                        k_z = n_z * DELTA_K;
     
                        k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);
     
                        // now go through the k bins and update
                        ct = 0; 
                        k_floor = 0; 
                        k_ceil = k_first_bin_ceil;
                        while (k_ceil < k_max){
                            // check if we fal in this bin
                            if ((k_mag>=k_floor) && (k_mag < k_ceil)){
                                in_bin_ct[ct]++;
                                p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T_LC[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
                                // note the 1/VOLUME factor, which turns this into a power density in k-space
     
                                k_ave[ct] += k_mag;
                                break;
                            }    
     
                            ct++;
                            k_floor=k_ceil;
                            k_ceil*=k_factor;
                        }    
                    }    
                }    
            }
        } // end looping through k box

    }
    else {

        // Co-eval box, so should sample the entire cube

        // now construct the power spectrum file
        for (n_x=0; n_x<HII_DIM; n_x++){
            if (n_x>HII_MIDDLE)
                k_x =(n_x-HII_DIM) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y=0; n_y<HII_DIM; n_y++){

                if (n_y>HII_MIDDLE)
                    k_y =(n_y-HII_DIM) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z=0; n_z<=HII_MIDDLE; n_z++){
                    k_z = n_z * DELTA_K;

                    k_mag = sqrt(k_x*k_x + k_y*k_y + k_z*k_z);

                    // now go through the k bins and update
                    ct = 0;
                    k_floor = 0;
                    k_ceil = k_first_bin_ceil;
                    while (k_ceil < k_max){
                        // check if we fal in this bin
                        if ((k_mag>=k_floor) && (k_mag < k_ceil)){
                            in_bin_ct[ct]++;
                            p_box[ct] += pow(k_mag,3)*pow(cabs(deldel_T_LC[HII_C_INDEX(n_x, n_y, n_z)]), 2)/(2.0*PI*PI*VOLUME);
                            // note the 1/VOLUME factor, which turns this into a power density in k-space

                            k_ave[ct] += k_mag;
                            break;
                        }

                        ct++;
                        k_floor=k_ceil;
                        k_ceil*=k_factor;
                    }
                }
            }
        } // end looping through k box
    }
}

/**** Arrays declared and used *****/

void init_21cmMC_HII_arrays() {
    
    int i,j;
    Overdense_spline_GL_low = (float*) calloc(Nlow,sizeof(float));
    Fcoll_spline_GL_low = (float*) calloc(Nlow,sizeof(float));
    second_derivs_low_GL = (float*) calloc(Nlow,sizeof(float));
    Overdense_spline_GL_high = (float*) calloc(Nhigh,sizeof(float));
    Fcoll_spline_GL_high = (float*) calloc(Nhigh,sizeof(float));
    second_derivs_high_GL = (float*) calloc(Nhigh,sizeof(float));
    
    deltax_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_unfiltered_original = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deltax_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    xe_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    xe_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deldel_T = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    deldel_T_LC = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    deltax = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    Fcoll = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
#ifdef MINI_HALO
    log10_Mmin_unfiltered      = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    log10_Mmin_filtered        = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    log10_Mmin_MINI_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    log10_Mmin_MINI_filtered   = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    Fcoll_MINI = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
#endif
    xH = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    v = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    vel_gradient = (float *) calloc(HII_TOT_FFT_NUM_PIXELS,sizeof(float));
    delta_T = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    
    x_pos = (float*) calloc(N_RSD_STEPS,sizeof(float));
    x_pos_offset = (float*) calloc(N_RSD_STEPS,sizeof(float));
    delta_T_RSD_LOS = (float*) calloc(HII_DIM,sizeof(float));
    
    xi_low = (float*) calloc((NGLlow+1),sizeof(float));
    wi_low = (float*) calloc((NGLlow+1),sizeof(float));
    
    xi_high = (float*) calloc((NGLhigh+1),sizeof(float));
    wi_high = (float*) calloc((NGLhigh+1),sizeof(float));

    if(USE_MASS_DEPENDENT_ZETA) {
        xi_SFR = (float*) calloc((NGL_SFR+1),sizeof(float));
        wi_SFR = (float*) calloc((NGL_SFR+1),sizeof(float));

        log10_overdense_spline_SFR = (double*) calloc(NSFR_low,sizeof(double));
        Overdense_spline_SFR = (float*) calloc(NSFR_high,sizeof(float));
        
#ifndef MINI_HALO
        log10_Fcoll_spline_SFR = (double*) calloc(NSFR_low,sizeof(double));
        Fcoll_spline_SFR = (double*) calloc(NSFR_high,sizeof(float));
#endif
        second_derivs_SFR = (float*) calloc(NSFR_high,sizeof(float));
    }

    
    k_factor = 1.35;
    k_first_bin_ceil = DELTA_K;
    k_max = DELTA_K*HII_DIM;
    // initialize arrays
    // ghetto counting (lookup how to do logs of arbitrary bases in c...)
    NUM_BINS = 0;
    k_floor = 0;
    k_ceil = k_first_bin_ceil;
    while (k_ceil < k_max){
        NUM_BINS++;
        k_floor=k_ceil;
        k_ceil*=k_factor;
    }
    
    p_box = (double*) calloc(NUM_BINS,sizeof(double));
    k_ave = (double*) calloc(NUM_BINS,sizeof(double));
    in_bin_ct = (unsigned long long *)calloc(NUM_BINS,sizeof(unsigned long long));
}

void init_21cmMC_Ts_arrays() {
    
    int i,j;
    
    box = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    unfiltered_box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    
    Tk_box = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    x_e_box = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    Ts = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));
    
    inverse_diff = (float*) calloc(x_int_NXHII,sizeof(float));
    
    zpp_growth = (float *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    
    //for continuously run HII calc
#ifdef MINI_HALO
    log10_Fcoll_spline_SFR = (double*) calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
    Fcoll_spline_SFR = (float*) calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));

    log10_Fcoll_spline_SFR_MINI = (double*) calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
    Fcoll_spline_SFR_MINI = (float*) calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));

    for (i=0;i<NSFR_low*LOG10MTURN_NUM;i++){
        log10_Fcoll_spline_SFR[i] = 0.;
        log10_Fcoll_spline_SFR_MINI[i] = 0.;
    }
    for (i=0;i<NSFR_high*LOG10MTURN_NUM;i++){
        Fcoll_spline_SFR[i] = 0.;
        Fcoll_spline_SFR_MINI[i] = 0.;
    }
#endif

    if (USE_MASS_DEPENDENT_ZETA) {
        
        SFR_timescale_factor = (float *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
#ifdef MINI_HALO
        Mturn_interp_table = (double *)calloc(LOG10MTURN_NUM, sizeof(double));
        for (i=0; i <LOG10MTURN_NUM; i++){
          Mturn_interp_table[i] = pow(10., LOG10MTURN_MIN + (double)i*LOG10MTURN_INT);
        }
#endif
        
        for (i=0; i < NUM_FILTER_STEPS_FOR_Ts; i++){
          FcollLow_zpp_spline_acc[i] = gsl_interp_accel_alloc ();
          FcollLow_zpp_spline[i] = gsl_spline_alloc (gsl_interp_cspline, NSFR_low);

          second_derivs_Fcoll_zpp[i] = (float*) calloc(NSFR_high,sizeof(float));

#ifdef MINI_HALO
          FcollLow_zpp_spline_acc_MINI[i] = gsl_interp_accel_alloc ();
          FcollLow_zpp_spline_MINI[i] = gsl_spline_alloc (gsl_interp_cspline, NSFR_low);

          second_derivs_Fcoll_zpp_MINI[i] = (float*) calloc(NSFR_high,sizeof(float));
#endif
        }

        xi_SFR_Xray = (float*) calloc((NGL_SFR+1),sizeof(float));
        wi_SFR_Xray = (float*) calloc((NGL_SFR+1),sizeof(float));

        zpp_interp_table = (float*) calloc(zpp_interp_points_SFR, sizeof(float));

        redshift_interp_table = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts*Nsteps_zp, sizeof(float)); // New
        growth_interp_table = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts*N_USER_REDSHIFT, sizeof(float)); // New
#ifdef MINI_HALO
        Mcrit_atom_interp_table = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts*N_USER_REDSHIFT, sizeof(float)); // New
#endif

        overdense_Xray_low_table = (float*) calloc(NSFR_low,sizeof(float));
        log10_Fcollz_SFR_Xray_low_table = (double ***)calloc(N_USER_REDSHIFT,sizeof(double **)); //New
#ifdef MINI_HALO
        log10_Fcollz_SFR_Xray_low_table_MINI = (double ***)calloc(N_USER_REDSHIFT,sizeof(double **)); //New
#endif
        for(i=0;i<N_USER_REDSHIFT;i++){  // New
            log10_Fcollz_SFR_Xray_low_table[i] = (double **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
#ifdef MINI_HALO
            log10_Fcollz_SFR_Xray_low_table_MINI[i] = (double **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
#endif
            for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
                log10_Fcollz_SFR_Xray_low_table[i][j] = (double *)calloc(NSFR_low,sizeof(double));
#ifdef MINI_HALO
                log10_Fcollz_SFR_Xray_low_table_MINI[i][j] = (double *)calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
#endif
            }
        }

        Overdense_Xray_high_table = (float*) calloc(NSFR_high,sizeof(float));
        Fcollz_SFR_Xray_high_table = (float ***)calloc(N_USER_REDSHIFT,sizeof(float **)); //New
#ifdef MINI_HALO
        Fcollz_SFR_Xray_high_table_MINI = (float ***)calloc(N_USER_REDSHIFT,sizeof(float **)); //New
#endif
        for(i=0;i<N_USER_REDSHIFT;i++){  // New
            Fcollz_SFR_Xray_high_table[i] = (float **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
#ifdef MINI_HALO
            Fcollz_SFR_Xray_high_table_MINI[i] = (float **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
#endif
            for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
                Fcollz_SFR_Xray_high_table[i][j] = (float *)calloc(NSFR_high,sizeof(float));
#ifdef MINI_HALO
                Fcollz_SFR_Xray_high_table_MINI[i][j] = (float *)calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));
#endif
            }
        }
        
        SFR_for_integrals_Rct = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        dxheat_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dxion_source_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dxlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dstarlya_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
#ifdef MINI_HALO
        dstarlyLW_dt_box = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));

        SFR_for_integrals_Rct_MINI = (float *) calloc(HII_TOT_NUM_PIXELS,sizeof(float));

        dxheat_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dxion_source_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dxlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dstarlya_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
        dstarlyLW_dt_box_MINI = (double *) calloc(HII_TOT_NUM_PIXELS,sizeof(double));
#endif
        
        delNL0 = (float **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS,sizeof(float));
        }

        m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS,sizeof(int));
        inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS,sizeof(float));
        
    }
    else {

        fcoll_R_grid = (double ***)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
        dfcoll_dz_grid = (double ***)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double **));
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            fcoll_R_grid[i] = (double **)calloc(zpp_interp_points,sizeof(double *)); 
            dfcoll_dz_grid[i] = (double **)calloc(zpp_interp_points,sizeof(double *)); 
            for(j=0;j<zpp_interp_points;j++) {
                fcoll_R_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
                dfcoll_dz_grid[i][j] = (double *)calloc(dens_Ninterp,sizeof(double));
            }    
        }    

        grid_dens = (double **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            grid_dens[i] = (double *)calloc(dens_Ninterp,sizeof(double));
        }

        density_gridpoints = (double **)calloc(dens_Ninterp,sizeof(double *));
        for(i=0;i<dens_Ninterp;i++) {
            density_gridpoints[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }
        ST_over_PS_arg_grid = (double *)calloc(zpp_interp_points,sizeof(double));

        dens_grid_int_vals = (short **)calloc(HII_TOT_NUM_PIXELS,sizeof(short *));
        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            dens_grid_int_vals[i] = (short *)calloc((float)NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
        }

        Sigma_Tmin_grid = (double *)calloc(zpp_interp_points,sizeof(double));

        fcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
        fcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
        dfcoll_interp1 = (double **)calloc(dens_Ninterp,sizeof(double *));
        dfcoll_interp2 = (double **)calloc(dens_Ninterp,sizeof(double *));
        for(i=0;i<dens_Ninterp;i++) {
            fcoll_interp1[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            fcoll_interp2[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dfcoll_interp1[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
            dfcoll_interp2[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        }

        delNL0_bw = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        delNL0_Offset = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        delNL0_LL = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        delNL0_UL = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        delNL0_ibw = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        log10delNL0_diff = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        log10delNL0_diff_UL = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        
        delNL0_rev = (float **)calloc(HII_TOT_NUM_PIXELS,sizeof(float *));
        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            delNL0_rev[i] = (float *)calloc((float)NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
        }

    }
    
    fcoll_R_array = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
   
    zpp_edge = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sigma_atR = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sigma_Tmin = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    ST_over_PS = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sum_lyn = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
#ifdef MINI_HALO
    ST_over_PS_MINI = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sum_lyn_MINI = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sum_lyLWn = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    sum_lyLWn_MINI = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
#endif

    zpp_for_evolve_list = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    R_values = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    SingleVal_float = (float*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
   
    freq_int_heat_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_ion_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_lya_tbl = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_heat_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_ion_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_lya_tbl_diff = (double **)calloc(x_int_NXHII,sizeof(double *));
    for(i=0;i<x_int_NXHII;i++) {
        freq_int_heat_tbl[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_ion_tbl[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_lya_tbl[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_heat_tbl_diff[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_ion_tbl_diff[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_lya_tbl_diff[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    }

    dstarlya_dt_prefactor = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
#ifdef MINI_HALO
    dstarlyLW_dt_prefactor = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));

    freq_int_heat_tbl_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_ion_tbl_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_lya_tbl_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_heat_tbl_diff_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_ion_tbl_diff_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    freq_int_lya_tbl_diff_MINI = (double **)calloc(x_int_NXHII,sizeof(double *));
    for(i=0;i<x_int_NXHII;i++) {
        freq_int_heat_tbl_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_ion_tbl_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_lya_tbl_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_heat_tbl_diff_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_ion_tbl_diff_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
        freq_int_lya_tbl_diff_MINI[i] = (double *)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    }

    dstarlya_dt_prefactor_MINI = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    dstarlyLW_dt_prefactor_MINI = (double*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
#endif
    
    SingleVal_int = (short int*) calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(short));
}

void init_LF_arrays() { // New in v1.4
    
    // allocate memory for arrays of halo mass and UV magnitude
    lnMhalo_param = (double*) calloc((NBINS_LF),sizeof(double));
    Muv_param = (double*) calloc((NBINS_LF),sizeof(double));
    log10phi = (double*) calloc((NBINS_LF),sizeof(double));
    Mhalo_param = (double*) calloc((NBINS_LF),sizeof(double));

    LF_spline_acc = gsl_interp_accel_alloc();
    LF_spline = gsl_spline_alloc(gsl_interp_cspline, NBINS_LF);
#ifdef MINI_HALO
    Mhalo_param_MINI = (double*) calloc((NBINS_LF),sizeof(double));
    Muv_param_MINI = (double*) calloc((NBINS_LF),sizeof(double));
    log10phi_MINI = (double*) calloc((NBINS_LF),sizeof(double));
    LF_spline_acc_MINI = gsl_interp_accel_alloc();
    LF2_spline_acc_MINI = gsl_interp_accel_alloc();
    LF3_spline_acc_MINI = gsl_interp_accel_alloc();
    LF_spline_MINI = gsl_spline_alloc(gsl_interp_cspline, NBINS_LF);
    LF2_spline_MINI = gsl_spline_alloc(gsl_interp_cspline, NBINS_LF);
    LF3_spline_MINI = gsl_spline_alloc(gsl_interp_cspline, NBINS_LF);
#endif
}

void destroy_21cmMC_HII_arrays(int skip_deallocate) {
    
    fftwf_free(deltax_unfiltered);
    fftwf_free(deltax_unfiltered_original);
    fftwf_free(deltax_filtered);
    fftwf_free(deldel_T);
    fftwf_free(deldel_T_LC);
    fftwf_free(xe_unfiltered);
    fftwf_free(xe_filtered);
    
    free(xH);
    free(deltax);
    free(Fcoll);
#ifdef MINI_HALO
    free(log10_Mmin_unfiltered);
    free(log10_Mmin_filtered);
    free(log10_Mmin_MINI_unfiltered);
    free(log10_Mmin_MINI_filtered);
    free(Fcoll_MINI);
#endif
    free(delta_T);
    free(v);
    free(vel_gradient);
    free(p_box);
    free(k_ave);
    free(in_bin_ct);
    
    free(x_pos);
    free(x_pos_offset);
    free(delta_T_RSD_LOS);
    
    free(Overdense_spline_GL_low);
    free(Fcoll_spline_GL_low);
    free(second_derivs_low_GL);
    free(Overdense_spline_GL_high);
    free(Fcoll_spline_GL_high);
    free(second_derivs_high_GL);

    free(xi_low);
    free(wi_low);
    
    free(xi_high);
    free(wi_high);

    if(USE_MASS_DEPENDENT_ZETA) {
        free(xi_SFR);
        free(wi_SFR);
#ifndef MINI_HALO
        free(log10_Fcoll_spline_SFR);
        free(Fcoll_spline_SFR);
#endif
        free(log10_overdense_spline_SFR);
        free(Overdense_spline_SFR);
        free(second_derivs_SFR);
        
        gsl_interp_accel_free(FcollLow_spline_acc);
        gsl_spline_free(FcollLow_spline);

    }
    
    if(skip_deallocate!=1) {
        free(Mass_Spline);
        free(Sigma_Spline);
        free(dSigmadm_Spline);
        free(second_derivs_sigma);
        free(second_derivs_dsigma);
    }
}

void destroy_21cmMC_Ts_arrays() {
    
    int i,j;
    
    fftwf_free(box);
    fftwf_free(unfiltered_box);
    
    free(Tk_box); free(x_e_box); free(Ts);

#ifdef MINI_HALO
    free(log10_Fcoll_spline_SFR);
    free(Fcoll_spline_SFR);
    free(log10_Fcoll_spline_SFR_MINI);
    free(Fcoll_spline_SFR_MINI);
#endif

    if (USE_MASS_DEPENDENT_ZETA) {
        free(xi_SFR_Xray);
        free(wi_SFR_Xray);

        free(SFR_timescale_factor);
#ifdef MINI_HALO
        free(Mturn_interp_table);
#endif
        
        free(zpp_interp_table);
        free(redshift_interp_table);
        free(growth_interp_table);
        
        for (i=0; i < NUM_FILTER_STEPS_FOR_Ts; i++){
              gsl_spline_free (FcollLow_zpp_spline[i]);
              gsl_interp_accel_free (FcollLow_zpp_spline_acc[i]);
              free(second_derivs_Fcoll_zpp[i]);
#ifdef MINI_HALO
              gsl_spline_free (FcollLow_zpp_spline_MINI[i]);
              gsl_interp_accel_free (FcollLow_zpp_spline_acc_MINI[i]);
              free(second_derivs_Fcoll_zpp_MINI[i]);
#endif
        }

        
        free(overdense_Xray_low_table);
        free(Overdense_Xray_high_table);
        
        
        for(i=0;i<N_USER_REDSHIFT;i++){
            for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
                free(log10_Fcollz_SFR_Xray_low_table[i][j]);
                free(Fcollz_SFR_Xray_high_table[i][j]);
            }
            free(log10_Fcollz_SFR_Xray_low_table[i]);
            free(Fcollz_SFR_Xray_high_table[i]);
        }
        free(log10_Fcollz_SFR_Xray_low_table);
        free(Fcollz_SFR_Xray_high_table);

        free(Mass_Spline_Xray);
        free(Sigma_Spline_Xray);
        free(dSigmadm_Spline_Xray);
        free(second_derivs_sigma_Xray);
        free(second_derivs_dsigma_Xray);
        
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            free(delNL0[i]);
        }
        free(delNL0);
        
        free(SFR_for_integrals_Rct);
        free(dxheat_dt_box);
        free(dxion_source_dt_box);
        free(dxlya_dt_box);
        free(dstarlya_dt_box);
#ifdef MINI_HALO
        free(dstarlyLW_dt_box);
        free(SFR_for_integrals_Rct_MINI);
        free(dxheat_dt_box_MINI);
        free(dxion_source_dt_box_MINI);
        free(dxlya_dt_box_MINI);
        free(dstarlya_dt_box_MINI);
        free(dstarlyLW_dt_box_MINI);
#endif 
        
        free(m_xHII_low_box);
        free(inverse_val_box);
        
        
        
        
    }
    else {
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            for(j=0;j<zpp_interp_points;j++) {
                free(fcoll_R_grid[i][j]);
                free(dfcoll_dz_grid[i][j]);
            }
            free(fcoll_R_grid[i]);
            free(dfcoll_dz_grid[i]);
        }
        free(fcoll_R_grid);
        free(dfcoll_dz_grid);
    
        for(i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++) {
            free(grid_dens[i]);
        }
        free(grid_dens);
    
        for(i=0;i<dens_Ninterp;i++) {
            free(density_gridpoints[i]);
        }
        free(density_gridpoints);
    
        free(ST_over_PS_arg_grid);
        free(Sigma_Tmin_grid);

        for(i=0;i<dens_Ninterp;i++) {
            free(fcoll_interp1[i]);
            free(fcoll_interp2[i]);
            free(dfcoll_interp1[i]);
            free(dfcoll_interp2[i]);
        }
        free(fcoll_interp1);
        free(fcoll_interp2);
        free(dfcoll_interp1);
        free(dfcoll_interp2);

        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            free(dens_grid_int_vals[i]);
        }
        free(dens_grid_int_vals);

        free(delNL0_bw);
        free(delNL0_Offset);
        free(delNL0_LL);
        free(delNL0_UL);
        free(delNL0_ibw);
        free(log10delNL0_diff);
        free(log10delNL0_diff_UL);

        free_FcollTable();
        
        for(i=0;i<HII_TOT_NUM_PIXELS;i++) {
            free(delNL0_rev[i]);
        }
        free(delNL0_rev);
    }
    
    free(fcoll_R_array);
    free(zpp_growth);
    free(inverse_diff);
    

    free(zpp_for_evolve_list);
    free(R_values);
    free(SingleVal_int);
    free(SingleVal_float);
    free(dstarlya_dt_prefactor);
#ifdef MINI_HALO
    free(dstarlya_dt_prefactor_MINI);
    free(dstarlyLW_dt_prefactor);
    free(dstarlyLW_dt_prefactor_MINI);
#endif

    free(zpp_edge);
    free(sigma_atR);
    free(sigma_Tmin);
    free(ST_over_PS);
    free(sum_lyn);
    
    for(i=0;i<x_int_NXHII;i++) {
        free(freq_int_heat_tbl[i]);
        free(freq_int_ion_tbl[i]);
        free(freq_int_lya_tbl[i]);
        free(freq_int_heat_tbl_diff[i]);
        free(freq_int_ion_tbl_diff[i]);
        free(freq_int_lya_tbl_diff[i]);
    }
    free(freq_int_heat_tbl);
    free(freq_int_ion_tbl);
    free(freq_int_lya_tbl);
    free(freq_int_heat_tbl_diff);
    free(freq_int_ion_tbl_diff);
    free(freq_int_lya_tbl_diff);
    
#ifdef MINI_HALO
    free(ST_over_PS_MINI);
    free(sum_lyn_MINI);
    free(sum_lyLWn);
    free(sum_lyLWn_MINI);
    
    for(i=0;i<x_int_NXHII;i++) {
        free(freq_int_heat_tbl_MINI[i]);
        free(freq_int_ion_tbl_MINI[i]);
        free(freq_int_lya_tbl_MINI[i]);
        free(freq_int_heat_tbl_diff_MINI[i]);
        free(freq_int_ion_tbl_diff_MINI[i]);
        free(freq_int_lya_tbl_diff_MINI[i]);
    }
    free(freq_int_heat_tbl_MINI);
    free(freq_int_ion_tbl_MINI);
    free(freq_int_lya_tbl_MINI);
    free(freq_int_heat_tbl_diff_MINI);
    free(freq_int_ion_tbl_diff_MINI);
    free(freq_int_lya_tbl_diff_MINI);
#endif
}

void destroy_LF_arrays() { // New in v1.4
    // free initialise of the interpolation
    gsl_interp_accel_free(LF_spline_acc);
    gsl_spline_free(LF_spline);

    // free memory allocation
    free(lnMhalo_param);
    free(Muv_param);
    free(log10phi);
    free(Mhalo_param);
#ifdef MINI_HALO
    free(Mhalo_param_MINI);
    gsl_interp_accel_free(LF_spline_acc_MINI);
    gsl_interp_accel_free(LF2_spline_acc_MINI);
    gsl_interp_accel_free(LF3_spline_acc_MINI);
    gsl_spline_free(LF_spline_MINI);
    gsl_spline_free(LF2_spline_MINI);
    gsl_spline_free(LF3_spline_MINI);
    free(Muv_param_MINI);
    free(log10phi_MINI);
#endif
}
