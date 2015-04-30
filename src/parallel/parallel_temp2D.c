/********************************************************************************
* FILE: parallel_temp2D.c
* DESCRIPTION:  
*   simplified two-dimensional temperature equation domain decomposition parallelised.  
* Last Revised: 11/12/14 Nicholas Betsworth
********************************************************************************/
#include <mpi.h>
#include <stdio.h>

// The rank of the master process
#define MASTER 0

#define XDIM 100
#define YDIM 100

// Determines whether the data is saved to a file after each timestep
#define SAVE_EACH_TIMESTEP 1

struct Params
{ 
	double cx; /* sampling size along X */
	double cy; /* sampling size along Y */
	int nts; /* timesteps */
}params = {0.1,0.1,50}; 

void initdata(int nx, int ny, double *u1);
void update(int nx, int ny, double *u1, double *u2);
void prtdata(int nx, int ny, int ts, double *u1, char* fname);
void placeinside(int nx, int ny, double *u1, double *u2);

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	double start_time, end_time;
	int num_procs, rank;
	// Iterators
	int i, j, n;
	// Stores the current time step
	int ts = 0;
	// Stores the initial data
	double array[XDIM][YDIM];
	// Stores data from the intermediate calculations
	// This array excludes the border values
	double receive_array[XDIM - 2][YDIM - 2];
	// Stores the final values after a calculation before saving to file
	double final_array[XDIM][YDIM];
	
	// Load in the number of processors
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	// Load in the rank of this processor
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Calculate the number of processors that will complete work
	// We remove the master process from this count as it will not
	// Be performing the same calculations as the rest of the processes
	int num_workers = num_procs;
	
	// Calculate how many rows each process will be calculating for
	// We subtract 2 as these are the boundary rows,
	// for which we will not be performing any calculations
	int num_rows = (XDIM - 2) / num_workers;
	// Calculate how many rows are not evenly divisible (we will give these to the last node)
	int leftover_rows = (XDIM - 2) % num_workers;
	
	// If this is the master process
	if(rank == MASTER)
	{
		// Initialise the array
		initdata(XDIM, YDIM, &array[0][0]);
		printf("%d rows leftover\n", leftover_rows);
		
		/* Output the initial array for testing purposes
		printf("Array initialized to:\n");
		for(i = 0; i < XDIM; i++) {
			for(j = 0; j < YDIM; j++) {
				printf("%f ", array[i][j]);
			}
			
			printf("\n");
		}*/
		
		// Initialise the borders of final_array using our initial data
		for(i = 0; i < XDIM; i++)
		{
			final_array[i][0] = array[i][0];
			final_array[i][YDIM - 1] = array[i][YDIM - 1];
		}
		
		for(j = 0; j < YDIM; j++)
		{
			final_array[0][j] = array[0][j];
			final_array[XDIM - 1][j] = array[XDIM - 1][j];
		}
	}
	
	// The size of array each node will receive
	int scatter_sizes[num_workers];
	// The displacement of elements being sent from the root node
	int scatter_displs[num_workers];
	// The size of the array each node will send back
	int gather_sizes[num_workers];
	// The displacement of the data being sent to the root node
	int gather_displs[num_workers];
	
	// Calculate the parameters for scatter
	for(n = 0; n < num_workers; n++)
	{
		scatter_displs[n] = n * (num_rows * YDIM);
		scatter_sizes[n] = (num_rows + 2) * YDIM;
	}
	
	// Give the last worker the left over rows
	scatter_sizes[num_workers - 1] += leftover_rows * YDIM;
	
	int n_array_size, n_ghost_elements, n_rows;
	int current_displ = 0;
	// Calculate the parameters for gather
	for(n = 0; n < num_workers; n++)
	{
		// Get the array size that was provided to this worker by the root node
		n_array_size = scatter_sizes[n];
		// Work out the number of rows in the original data
		n_rows = n_array_size / YDIM;
		// Calculate the number of ghost elements in the array
		// We have two ghost cells per row (left and right most columns)
		// And we have a row above and below which are all ghost cells
		// Subtract 4 cells, as we count 4 of the cells twice in the stated calculation
		n_ghost_elements = (n_rows * 2) + (YDIM * 2)  - 4;
		
		// The size of the array we will be sending back to the root node
		// Is the original size - the ghost elements around the outside
		gather_sizes[n] = n_array_size - n_ghost_elements;
		// The displacements of the data from each of the ranks on the root node
		gather_displs[n] = current_displ;
		current_displ += gather_sizes[n];
		
		/*if(rank == MASTER)
		{
			printf("Proc: %d, Gather size: %d, Scatter size: %d, Gather displ: %d\n", n, gather_sizes[n], scatter_sizes[n], gather_displs[n]);
		}*/
	}
	
	// Store the size of the array we will be receiving from root
	int my_array_size = scatter_sizes[rank];
	// Calculate the number of rows we will receive
	int row_count = my_array_size / YDIM;
	
	// Stores the elements we receive
	double elements[my_array_size];
	// Stores the results of our calculation
	double updated_elements[gather_sizes[rank]];
	
	// Start the timer on the root node when all processes reach this point
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == MASTER)
		start_time = MPI_Wtime();
		
	// Spread out the data from the root node to the slave nodes (this includes the root)
	MPI_Scatterv(&array[0][0], scatter_sizes, scatter_displs, MPI_DOUBLE, elements, my_array_size, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	
	// Update once outside the loop to avoid unnecessary swapping and data transfer on the last iteration
	update(row_count, YDIM, elements, updated_elements);
	ts++;
	
	// Perform the calculations for the required number of time steps
	while(ts < params.nts)
	{
		// If the program is set to save to a new file at each timestep
		if(SAVE_EACH_TIMESTEP)
		{
			// Gather all of the calculated data at the root node
			MPI_Gatherv(updated_elements, gather_sizes[rank], MPI_DOUBLE, receive_array, gather_sizes, gather_displs, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			if(rank == MASTER)
			{
				// Place the un-bordered array inside the array with the border data
				placeinside(XDIM - 2, YDIM - 2, &final_array[0][0], &receive_array[0][0]);
				// Save the data to a file
				prtdata(XDIM, YDIM, ts, &final_array[0][0], "final_data");
			}
		}
		
		// Swap updated_elements in to elements
		for(i = 0; i < row_count - 2; i++)
		{
			for(j = 0; j < YDIM - 2; j++)
			{
				elements[(i + 1) * YDIM + (j + 1)] = updated_elements[i * (YDIM - 2) + j];
			}
		}
		// Send the updated ghost cells to the above/below processors
		// First send all of the data to the processors above itself
		// If the rank is greater than zero then we have data to send to the above process
		MPI_Status status;
		if(rank > 0 && rank < num_procs - 1)
		{
			MPI_Sendrecv(&updated_elements, YDIM - 2, MPI_DOUBLE, rank - 1, 0, &elements[YDIM * (row_count - 1) + 1], YDIM - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
		}
		else if(rank == 0)
		{
			MPI_Recv(&elements[YDIM * (row_count - 1) + 1], YDIM - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
		} else if(rank == num_procs - 1)
		{
			MPI_Send(&updated_elements, YDIM - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
		}
		
		// Now send all of the data to the processors below
		// The start of our index for the send buffer is the original row count (including ghost cells)
		// minus 2 because updated_elements does not contain ghost cells, minus another 1 to place us at the
		// start of the last row
		int send_index = (row_count - (2 + 1)) * (YDIM - 2);
		if(rank > 0 && rank < num_procs - 1)
		{
			MPI_Sendrecv(&updated_elements[send_index], YDIM - 2, MPI_DOUBLE, rank + 1, 1, &elements[1], YDIM - 2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
		}
		else if(rank == 0)
		{
			MPI_Send(&updated_elements[send_index], YDIM - 2, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
		}
		else if(rank == num_procs - 1)
		{
			MPI_Recv(&elements[1], YDIM - 2, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
		}
		
		// Update the data by another time step
		update(row_count, YDIM, elements, updated_elements);
		ts++;
	}
	
	// Combine the data that all processors have calculated back in to a single array
	MPI_Gatherv(updated_elements, gather_sizes[rank], MPI_DOUBLE, receive_array, gather_sizes, gather_displs, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	
	if(rank == MASTER)
	{
		// Store the end time here to discount the time taken to save to file
		end_time = MPI_Wtime();
		
		// Place the un-bordered array inside the array with the border data
		placeinside(XDIM - 2, YDIM - 2, &final_array[0][0], &receive_array[0][0]);
		// Save the data to a file
		prtdata(XDIM, YDIM, ts, &final_array[0][0], "final_data");
		
		printf("Time taken to calculate: %.3fs\n", end_time - start_time);
		/*printf("\nOutputting final array: \n");
		for(i = 0; i < XDIM; i++) {
			for(j = 0; j < YDIM; j++) {
				printf("%.1f ", final_array[i][j]);
			}
			
			printf("\n");
		}*/
	}
	
	MPI_Finalize();
	return 0;
}

/***
*  placeinside: places array u2 inside the border of array u1
				nx, ny refer to the width and height of u2
***/
void placeinside(int nx, int ny, double *u1, double *u2)
{
	// YDIM of the bordering array (u1)
	int ny2 = ny + 2;
	
	int ix, iy;
	for (ix = 0; ix < nx; ix++) {
		for (iy = 0; iy < ny; iy++) {
			u1[(ix + 1) * ny2 + iy + 1] = u2[ix * ny + iy];
		}
	}
}

/***
*  initdata: initializes old_u, timestep t=0
***/
void initdata(int nx, int ny, double *u1)
{
	int ix, iy;
	for (ix = 0; ix <= nx-1; ix++) 
		for (iy = 0; iy <= ny-1; iy++) 
			*(u1+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));

}

/***
*  update: computes new values for timestep t+delta_t
***/
void update(int nx, int ny, double *u1, double *u2)
{
	int ix, iy;

	for (ix = 1; ix <= nx-2; ix++) {
		for (iy = 1; iy <= ny-2; iy++) {
			// Changed so that we subtract 1 from ix and iy with respect
			// to the position of element in u2, as we do not have ghost cells
			// in u2. We also subtract 2 from ny, as our array is 2 elements short of ny
			*(u2+(ix - 1)*(ny - 2) +(iy - 1)) = *(u1+ix*ny+iy)  + 
				params.cx * (*(u1+(ix+1)*ny+iy) + *(u1+(ix-1)*ny+iy) - 
				2.0 * *(u1+ix*ny+iy)) +
				params.cy * (*(u1+ix*ny+iy+1) + *(u1+ix*ny+iy-1) - 
				2.0 * *(u1+ix*ny+iy));
		}
	}
}

/***
*  prtdata: generates a .csv file with data contained in parameter double* u1 
***/
void prtdata(int nx, int ny, int ts, double *u1, char* fname)
{
	int ix, iy;
	FILE *fp;
	// create a buffer for our filename, as we will be automatically generating it
	char filepath[64];
	// generate the filename with respect to the time step passed in (ts)
	sprintf(filepath, "%s_%d.dat", fname, ts);
	
	// write each of the values to the csv file
	fp = fopen(filepath, "w");
	for (iy = 0; iy < ny; iy++)
	{
		for (ix = 0; ix < nx; ix++)
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
	
	// close the file when we have finished writing to it
	fclose(fp);
	
	printf(" %s\n",filepath);
}