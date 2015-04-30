/****************************************************************************
* FILE: serial_temp2D.c
* DESCRIPTION:  
*   simplified two-dimensional temperature equation domain decomposition.  
* Last Revised: 18/11/14 Nicholas Betsworth
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define XDIM 100
#define YDIM 100

/***
	Params data structure stores parameters passed in by the user
	Including:
		cx:		X sampling size
		cy:		Y sampling size
		nts: 	number of time steps
*/
struct Params
{ 
	double cx; /* sampling size along X */
	double cy; /* sampling size along Y */
	int nts; /* timesteps */
}; 


struct Params P; /* stores the parameters passed in by the user */

void initdata(double *u1, char* fname);
void copyboundaries(int nx, int ny, double *u1, double *u2);
void update(int nx, int ny, double *u1, double *u2);
void prtdata(int nx, int ny, int ts, double *u1, char* fname);
void printarray(int nx, int ny, double *u1);


int main(int argc, char *argv[])
{
	/* Ensure the correct number of parameters has been provided */
	if(argc < 5)
	{
		perror("Incorrect Usage. Expected: temp_dist2D Cx Cy nts csv_file.csv\n ");
		exit(1);
	}
	
	/* Read in grid sampling parameters and timesteps and initialize struct Params */
	P.cx = atof(argv[1]);
	P.cy = atof(argv[2]);
	P.nts = atoi(argv[3]);
	/* Read in filename of csv file */
	char* fname = argv[4];
	
	/* Perform some quick validation on the parameters */
	if(P.cx <= 0.0 || P.cy <= 0.0)
	{
		perror(" Error: Cx and Cy must be greater than 0.\n ");
		exit(1);
	}
	
	if(P.nts < 0)
	{
		perror(" Error: number of time steps must be at least 0.\n ");
		exit(1);
	}
	
	/* define the two arrays that will be used when calculating the new timesteps */
	double old_u[XDIM][YDIM];
	double new_u[XDIM][YDIM];
	double *old_array = &old_u[0][0]; /* Stores a pointer to the array with the initial values */
	double *new_array = &new_u[0][0]; /* Stores a pointer to the array which will store the values for the new timestep */
	double *temp_array; /* Temporary store when swapping old_array and new_array */
	
	int ix, iy, iz, ts; /* iterators */

	printf("Starting serial version of 2D temperature distribution example...\n");

	printf("Using [%d][%d] grid.\n", XDIM, YDIM);

	/* initialize grid from input file */
	printf("Initializing grid from input file..\n");

	initdata(&old_u[0][0], fname);
	
	/* copy boundary conditions from old_u to new_u */
	copyboundaries(XDIM, YDIM, &old_u[0][0], &new_u[0][0]);
	
	clock_t start, end;
	double runtime;
	
	/* iterate over all timesteps */
	printf("Iterating over %d time steps...\n", P.nts);
	
	start = clock(); /* start the timer for measuring performance */
	for(ts = 0; ts < P.nts; ts++)
	{
		/* Update the array with the next time step */
		update(XDIM, YDIM, old_array, new_array);
		
		/* swap the array pointers to save us having to transfer the data between arrays */
		temp_array = old_array;
		old_array = new_array;
		new_array = temp_array;
		
		/* output the new data to a numbered csv file */
		/* uncomment this to print an individual csv file for each time step*/ 
		//prtdata(XDIM, YDIM, ts, old_array, "final_data");
	}
	
	/* calculate and output the runtime of the timestep calculations */
	end = clock();
	runtime = (double)(end - start) / CLOCKS_PER_SEC;
	printf("%d timesteps calculated in %.3f seconds. \n", P.nts, runtime);
	
	printf("Done. Created output file(s) %d\n", ts);
	
	/* print the final data to a new csv file */
	/* this step is not necessary if you have prtdata commented out within the loop */
	prtdata(XDIM, YDIM, ts, old_array, "final_data");
}

/***
*  copyboundaries: copies the boundary values from array u1 to array u2
*	nx	The width of the array
*	ny 	The height of the array
*	u1	The original array
*	u2	New array to copy boundary values to
***/
void copyboundaries(int nx, int ny, double *u1, double *u2)
{
	int ix, iy;
	
	/* Loop width wise */
	for (ix = 0; ix < nx; ix++) 
	{
		/* Top row */
		u2[ix] = u1[ix];
		/* Bottom row */
		u2[ny * (nx - 1) + ix] = u1[ny * (nx - 1) + ix];
	}
	
	/* Loop height wise */
	/* Note: we leave out the top and bottom row as it has already been complete */
	for (iy = 1; iy < ny - 1; iy++) 
	{
		/* Left most column */
		u2[iy * nx] = u1[iy * nx];
		/* Right most column */
		u2[iy * nx + (nx - 1)] = u1[iy * nx  + (nx - 1)];
	}
}

/***
*  update: computes new values for timestep t+delta_t
*	nx	The width of the array
*	ny 	The height of the array
*	u1	Current values
*	u2	Array to store new values
***/
void update(int nx, int ny, double *u1, double *u2)
{
	int ix, iy; /* iterators */
	int k; /* temp store for the current key in the array */
	
	/* Iterate through the inside of the array (Leave a border of 1 element around the outside) */
	for(iy = 1; iy < ny - 1; iy++)
	{
		for(ix = 1; ix < nx - 1; ix++)
		{
			/* Store k as the key to array element [ix][iy] */
			k = iy * nx + ix;
			u2[k] = u1[k] + 
					P.cx * (u1[k + 1] + u1[k - 1] - 2 * u1[k]) +
					P.cy * (u1[k + nx] + u1[k - nx] - 2 * u1[k]);
		}
	}

}

/***
*  initdata: initializes old_u with data read from file specified at command line
***/
void initdata(double *u1, char* fname)
{
	int ix, iy; /* iterators */
	FILE* fp;

	fp = fopen(fname, "r");
	
	/* If the file was not successfully opened */
	if(fp == NULL)
	{
		perror("Error: unable to open specified file. Exiting.\n");
		exit(1);
	}
	
	const int buffer_size = 4096; /* the size of the buffer for each line being read in */
	char str[buffer_size]; /* read buffer */
	
	char *tok; /* used to read in each value */
	iy = 0;
	/* loops through each line in the csv file */
	while(fgets(str, buffer_size, fp) != NULL)
	{
		tok = strtok(str, ", "); /* split the line when a comma, or space operator is detected */
		
		for(ix = 0; ix < XDIM; ix++)
		{
			u1[ix + iy * XDIM] = atof(tok); /* convert the char input to a double and store it */
			tok = strtok(NULL, ","); /* read the next value in */
		}
		iy++;
	}
	
	/* if we read in less lines than the Y dimension specified, throw an error, as we have insufficient data */
	if(iy < YDIM)
	{
		perror("Error: end of file reached before all rows were filled in.\n");
		exit(1);
	}
	
	/* close the file as we have finished using it */
	fclose(fp);
	
}
/***
*  prtdata: generates a .csv file with data contained in parameter double* u1 
***/
void prtdata(int nx, int ny, int ts, double *u1, char* fname)
{
	int ix, iy;
	FILE *fp;
	/* create a buffer for our filename, as we will be automatically generating it */
	char filepath[64];
	/* generate the filename with respect to the time step passed in (ts) */
	sprintf(filepath, "%s_%d.csv", fname, ts);
	
	/* write each of the values to the csv file */
	fp = fopen(filepath, "w");
	for (iy = 0; iy < ny; iy++)
	{
		for (ix = 0; ix <= nx-1; ix++)
		{
			fprintf(fp, "%8.3f,", u1[ix + iy*nx]);
			if (ix != nx-1)
			{
				fprintf(fp, " ");
			}
			else {
				fprintf(fp, "\n");
			}
		}
	}
	
	/* close the file when we have finished writing to it */
	fclose(fp);
	
	printf(" %s\n",filepath);
}

/***
*  printarray: Prints the array u1 of dimensions [nx][ny], for debugging purposes
***/
void printarray(int nx, int ny, double *u1)
{
	int ix, iy; /* iterators */
	
	/* print each value to the console */
	for(iy = 0; iy < ny; iy++)
	{
		for(ix = 0; ix < nx; ix++)
		{
			printf("%f, ", u1[ix + iy * nx]);
		}
		
		/* start a new line at the end of each row */
		printf("\n");
	}
}