#include <boost/mpi.hpp>
#include <iostream>
#include <cstdlib>
#include <boost/serialization/vector.hpp>
namespace mpi = boost::mpi;

enum message_tags {msg_data_packet, msg_broadcast_data, msg_finished};

void arbeite_fuer_geld(mpi::communicator local, mpi::communicator world);
void sammel_geld_ein(mpi::communicator local, mpi::communicator world);

int main(int argc, char* argv[])
{
  mpi::environment env(argc,argv);
  mpi::communicator world;

  /* 2/3 der Prozesse werden arbeiter */
  bool ist_arbeiter = world.rank() < 2 * world.size() /3;
  mpi::communicator local = world.split(ist_arbeiter? 0:1);

  if(ist_arbeiter) arbeite_fuer_geld(local, world);
  else sammel_geld_ein(local, world);

  return 0;
}

void sammel_geld_ein(mpi::communicator local, mpi::communicator world)
{

	std::cout << "Prozess Nr. " << world.rank() << " ist in sammel_geld_ein" << std::endl;
	mpi::status msg = world.probe();

	std::string daten, daten_out;
	/* da die anderen über broadcast schicken können die Sammler diese Nachricht so empfangen */
	broadcast(world, daten, 0);
	std::cout << "An alle, ich (Prozess Nr. " << world.rank()
					<< ") habe eine Nachricht erhalten. " << daten << std::endl;

	daten_out = "okay, Nachricht ist angekommen";
	for (int dest = 0 ; dest < (world.size() - local.size()); ++dest)
	{
		world.send(dest, msg_broadcast_data, daten_out);
	}

// unwichtig
	if (msg.tag() == msg_data_packet)
	{
		 //Daten empfangen
		std::vector<int> data;
		/* durch den Status können wir jetzt auf id und tag zugreifen */
		world.recv(msg.source(), msg.tag(), data);

		/* sage allen sammlern das wir ein paar Daten haben? */
		for (int dest = 0 ; dest < local.size(); ++dest)
		{
			local.send(dest, msg_broadcast_data, msg.source());
		}
		std::cout << "An alle, ich (Prozess Nr. " << world.rank()
				<< ") habe eine Nachricht erhalten." << std::endl;
		/* weiterleiten den Nachricht */
		broadcast(local, data, 0);
	}

return;
}

void arbeite_fuer_geld(mpi::communicator local, mpi::communicator world)
{
	std::cout << "Prozess Nr. " << world.rank() << " ist in arbeite_fuer_geld" << std::endl;
	std::string daten, daten_in;
	daten = "ein kleiner test.";
	broadcast(world, daten, 0);

	std::cout << "Ich, prozess Nr. " << world.rank() << " habe gerade: '"
			<< daten << "' als nachricht an alle geschickt." << std::endl;

	mpi::status msg = world.probe();
	world.recv(msg.source(), msg.tag(), daten_in);

	std::cout << "Ich, prozess Nr. " << world.rank() << " habe gerade: '"
				<< daten_in << "' als nachricht Bestaetigung erhalten." << std::endl;


	return;
}



/* // Gather - einer fängt an und sagt er ist der Chef und dann schicken ihm alle eine Nachricht
#include <boost/mpi.hpp>
#include <vector>
#include <iostream>
#include <cstdlib>
namespace mpi = boost::mpi;

int main(int argc, char* argv[])
{
  mpi::environment env(argc,argv);
  mpi::communicator world;
  // zufällig Zahl mit der ID erstellen
  std::srand(time(0) + world.rank());
  int my_number = std::rand();

  // Chef-Prozess
  if (world.rank() == 0)
  {
	  // Vector wird dynamisch vergrößert
	  std::vector<int> all_numbers;
	  // Prozess 0 will alle nummern in all_numbers und gibt als id 0 an
	  gather(world, my_number, all_numbers, 0);
	  // In all_numbers stehen in Reihenfolge (nach id) die nummern drin
	  for (int proc = 0; proc < world.size(); ++proc)
	  {
		  std::cout << "Process #" << proc << " hat sich die Nummer " << all_numbers[proc]
		            << " vorgestellt." << std::endl;
	  }
  } else
  {
	  // alle andere Prozesse schicken ihre Nummer ans Gather mit der Id 0
	  gather(world, my_number, 0);
  }
}
*/


/*// Broadcast - Nachricht an alle
#include <boost/mpi.hpp>
#include <string>
#include <iostream>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;

int main(int argc, char* argv[])
{
  mpi::environment env(argc,argv);
  mpi::communicator world;
  
  std::string value;
  if (world.rank() == 0)   value = "Ich bins, die Nummer Null!!!";

  // Sende Nachricht an alle
  broadcast(world, value, 0);

  if (world.rank() != 0)
  {
	  std::cout << "Process #" << world.rank() << " hat mir geschrieben: " << value << std::endl;

  }
  return 0;
}
*/

/*// Request Objekt und Nebenläufigkeit
#include <boost/mpi.hpp>
#include <string>
#include <iostream>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;

int main(int argc, char* argv[])
{
  mpi::environment env(argc,argv);
  mpi::communicator world;

  if (world.rank() == 0)
  {
	  mpi::request reqs[4];
	  std::string msg1, msg2, out_msg1 = "0: Nummer Null an Nummer drei, bitte kommen!";
	  std::string out_msg2 = "0: Nummer Null an Nummer zwei, bitte kommen!";
		// Alle Befehle die ausgefuehrt werden sollen in Request Objekt
	  reqs[0] = world.isend( 1, 0, out_msg1);
	  reqs[1] = world.isend( 2, 1,  out_msg2);
	  reqs[2] = world.irecv( 1, 2, msg1);
	  reqs[3] = world.irecv( 2, 4, msg2);
	  // wait_all braucht zeiger auf erstes und letztes Element,
	  // dann wird gewartet bis alle abgehandelt wurden
	  mpi::wait_all(reqs, reqs + 4);
	  std::cout << msg1 << std::endl;
	  std::cout << msg2 << std::endl;
  }
  else if (world.rank() == 1)
  {
	  mpi::request reqs[4];
  	  std::string msg1, msg2, out_msg1 = "1: Nummer eins meldet sich zum Dienst!";
  	  std::string out_msg2 = "1: Ey, hat Nummer Null dich auch angeschrieben?!";
  	  reqs[0] = world.isend( 0, 2, out_msg1);
  	  reqs[1] = world.isend( 2, 3,  out_msg2);
  	  reqs[2] = world.irecv( 0, 0, msg1);
  	  reqs[3] = world.irecv( 2, 5, msg2);
  	  mpi::wait_all(reqs, reqs + 4);
  	  std::cout << msg1 << std::endl;
  	  std::cout << msg2 << std::endl;
  }
  else
  {
	  mpi::request reqs[4];
	  	  std::string msg1, msg2, out_msg1 = "2: Nummer drei meldet sich zum Dienst!";
	  	  std::string out_msg2 = "2: Ja leider, kein bock auf arbeiten!!!";
	  	  reqs[0] = world.isend( 0, 4, out_msg1);
	  	  reqs[1] = world.isend( 1, 5,  out_msg2);
	  	  reqs[2] = world.irecv( 0, 1, msg1);
	  	  reqs[3] = world.irecv( 1, 3, msg2);
	  	  mpi::wait_all(reqs, reqs + 4);
	  	  std::cout << msg1 << std::endl;
	  	  std::cout << msg2 << std::endl;
  }
return 0;
}
*/

/*
#include <boost/mpi.hpp>
#include <string>
#include <iostream>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;

int main(int argc, char* argv[])
{
  mpi::environment env(argc,argv);
  mpi::communicator world;

  if(world.rank() == 0)
  {
    world.send(1,0, std::string("ey du! Ja genau du...sagt mal was machste hier?"));
    std::string msg;
    world.recv( 1, 1, msg);
    
    std::cout << msg << " <-- Nachricht vom zweiten Prozess." << std::endl;
  }
  else 
  {
    std::string msg;
    world.recv(0, 0, msg);
    std::cout << msg << " <-- Nachricht vom ersten Prozess." << std::endl;
    std::cout.flush();
    world.send(0,1,std::string("Meinst du etwa mich? Was gibts?"));
  }

  return 0;
}
*/


/* erstes Programm, funktioniert
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
int main (int argc, char* argv[])
{
  std::cout << " neue Version " << std::endl;
  mpi::environment env(argc, argv);
  mpi::communicator world;
  std::cout << "I am process number " << world.rank() << " of " << world.size()
            << "." << std::endl;
  return 0;
}
*/
