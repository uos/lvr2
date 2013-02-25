/*
 * Programmierbeispiel der Stanfort Universität.
 *  http://www.slac.stanford.edu/comp/unix/farm/mpi.html
 */

#include <stdio.h>
#include <mpi.h>

int main (int argc , char *argv[]) {

	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	// Initialisiert die verbindung
	MPI_Init(&argc, &argv);

	//liefer die Anzahl an Prozessen
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// liefert die Id / Rang des Prozesses
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// liefert den namen des Processors (Rechner auf dem es ausgeführt wird)
	MPI_Get_processor_name(processor_name, &namelen);

	printf("Process %d on %s out of %d \n", rank, processor_name, numprocs);

	//VaterProcess oder spaeter Master-Process
	if (rank == 0)
	{
		mpi::request reqs[4];
		std::string msg1, msg2;
		msg1 = "Master an Untergebenen Nummer 1, alles in Ordnung dadrüben?";
		msg2 = "Master an Untergebenen Nummer 2, ich hoffe du hast das Klo geputzt sonst wirds übel!";

		reqs[0] = MPI_COMM_WORLD.MPI_Isend(msg1, sizeof(msg1), MPI_CHAR, 1, 0 );
		reqs[1] = MPI_COMM_WORLD.MPI_Isend(msg2, sizeof(msg1), MPI_CHAR, 2, 0 );
	}
	else
	{
		std::string recv[64];
		MPI_COMM_WORLD.recv(recv, sizeof(recv), MPI_CHAR, 0, 0);
		printf("Process %d on %s out of %d \n hat empfangen %s", rank, processor_name, numprocs, recv);

	}
	MPI_Finalize();

}
