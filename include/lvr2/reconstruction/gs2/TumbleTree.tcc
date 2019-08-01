//
// Created by patrick on 4/4/19.
//


namespace lvr2{


    //TODO: use REDBLACKTREE

    /**
     * Deletes the (subtree) beginning with cell c
     * @param c
     * @return
     */
    Cell* TumbleTree::makeEmpty(Cell* c)
    {
        if(c == NULL)
            return NULL;
        {
            makeEmpty(c->left);
            makeEmpty(c->right);
            delete c;
        }
        return NULL;
    }

    Cell* TumbleTree::makeCell(double sc, VertexHandle vH, Cell* left, Cell* right, Cell* parent, double alpha)
    {
        Cell* cell = new Cell();
        cell->parent = parent;
        cell->left = left;
        cell->right = right;
        cell->alpha = alpha;
        cell->signal_counter = sc;
        cell->duplicateMap.insert(vH, sc);

        return cell;
    }

    /*Cell* TumbleTree::insertIterative(double sc, VertexHandle vH)
    {

        Cell* newCell = new Cell();
        newCell->signal_counter = sc;
        newCell->duplicateMap.insert(vH,sc);
        newCell->left = NULL;
        newCell->right = NULL;
        newCell->parent = NULL;
        newCell->alpha = 1;

        if(root == NULL)
        {
            root = newCell;
            return root;
        }

        Cell* tmp = root;

        Cell* maintain = NULL;


        while(sc != tmp->signal_counter)
        {
            tmp->signal_counter *= tmp->alpha;
            if(tmp->left != NULL) tmp->left->alpha *= tmp->alpha;
            if(tmp->right != NULL) tmp->right->alpha *= tmp->alpha;
            tmp->alpha = 1;

            if(sc < tmp->signal_counter)
            {
                if(tmp->left == NULL)
                {
                    tmp->left = newCell;
                    newCell->parent = tmp;
                    return tmp->left;
                }

                tmp = tmp->left;
            }
            else if(sc > tmp->signal_counter)
            {
                if(tmp->right == NULL)
                {
                    tmp->right = newCell;
                    newCell->parent = tmp;
                    return tmp->right;
                }

                tmp = tmp->right;
            }
        }

        //found
        tmp->duplicateMap.insert(vH,sc);
        //cout << "Inserting a duplicate" << endl;
        return tmp;

    }*/

    Cell* TumbleTree::insert(Cell* c, double sc, VertexHandle vH)
    {
        //update the current cell's sc and propagate
        c->signal_counter *= c->alpha;
        if(c->left) c->left->alpha *= c->alpha;
        if(c->right) c->right->alpha *= c->alpha;
        c->alpha = 1;

        // if the sc is smaller than the current sc, go to the left subtree
        if(sc < c->signal_counter)
        {
            if(c->left)
                return insert(c->left, sc, vH);
            else //insert new cell, if the left subtree is empty
            {
                auto cell = makeCell(sc, vH, NULL, NULL, c, 1);
                c->left = cell;
                return cell;
            }
        }
        //if the sc is bigger than the cell's sc, go to the right subtree
        else if(sc > c->signal_counter)
        {
            if(c->right)
                return insert(c->right, sc, vH);
            else //insert new cell, if the right subtree is empty
            {
                auto cell = makeCell(sc, vH, NULL, NULL, c, 1);
                c->right = cell;
                return cell;
            }
        }
        else
        {
            c->duplicateMap.insert(vH, sc);
            return c;
        }
    }

    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits iterative??
    /*Cell* TumbleTree::removeTwo(double sc, VertexHandle vH, Cell* c, bool removeWhole, double alpha)
    {
        Cell* tmp;
        if(c == NULL)
        {
            c != root ? notDeleted++ : notDeleted = notDeleted-1+1;
            //std::cout << "  signal counter not found in TT" << endl;
            return NULL;
        }
        else{
            c->signal_counter *= c->alpha;
            if(c->left)c->left->alpha *= c->alpha;
            if(c->right)c->right->alpha *= c->alpha;
            c->alpha = 1;

            //cout << "Search sc: " << sc << " | Current sc: " << c->signal_counter << endl;
        }

        if(sc < c->signal_counter)
        {
            c->left = removeTwo(sc, vH, c->left, removeWhole, alpha);
            if(c->left) c->left->parent = c;
        }
        else if(sc > c->signal_counter) {
            c->right = removeTwo(sc, vH, c->right, removeWhole, alpha);
            if(c->right) c->right->parent = c;
        }
        else{
            //if there are two are more, just remove from the duplicate map
            if(c->duplicateMap.numValues() > 1 && !removeWhole){
                size_t numV = c->duplicateMap.numValues();
                c->duplicateMap.erase(vH); //erase index from duplicate map
                if(c->duplicateMap.numValues() == numV){
                    notDeleted++;
                    //std::cout << "  Not found in duplicate map..." << endl;
                }
                return c;
            }

            //if there is one or no child
            if(c->left == NULL)
            {
                tmp = c->right;
                delete c;
                return tmp;
            }
            else if(c->right == NULL)
            {
                tmp = c->left;
                delete c;
                return tmp;
            }
            //node with two children: get min of the right subtree (inorder successor of right subtree)
            tmp = findMin(c->right);

            //copy data from inorder successor
            c->signal_counter = tmp->signal_counter;
            c->duplicateMap.clear();
            //copy values from one dupilcate map to the other
            for(auto iter = tmp->duplicateMap.begin(); iter != tmp->duplicateMap.end();++iter)
            {
                c->duplicateMap.insert(*iter,tmp->signal_counter);
            }
            //remove inorder successor
            VertexHandle ret(0);
            c->right = removeTwo(tmp->signal_counter, ret, c->right, true, alpha); // no good...
            if(c->right) c->right->parent = c;

        }

        return c;
    }*/


    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits + 4  -  iterative??
    Cell* TumbleTree::remove(double sc, VertexHandle vH, Cell* c, bool removeWhole)
    {
        if(c == NULL){
            //cout << "Cell not found or tree null" << endl;
            return NULL;
        } //empty tree or not found

        c->signal_counter *= c->alpha;
        if(c->left) c->left->alpha *= c->alpha;
        if(c->right) c->right->alpha *= c->alpha;
        c->alpha = 1;

        //cout << "SC of interest: " << sc << " | Current sc: " << c->signal_counter << endl;

        if(sc < c->signal_counter)
        {
            //search left subtree, update parent
            c->left = remove(sc, vH, c->left, removeWhole);
            if(c->left)
            {
                c->left->parent = c;
            }

        }
        else if(sc > c->signal_counter)
        {
            //search right subtree, update parent
            c->right = remove(sc, vH, c->right, removeWhole);
            if(c->right)
            {
                c->right->parent = c;
            }
        }
        else
        {
            //found

            //if the duplicate map has one or more handles inside, erase from it
            if(c->duplicateMap.numValues() > 1 && !removeWhole)
            {
                long num = c->duplicateMap.numValues();
                c->duplicateMap.erase(vH);
                //if(num != c->duplicateMap.numValues()) cout << "found in duplicate Map" << endl;
            }
            //no children
            else if(!c->left && !c->right)
            {
                delete c; //delete leaf node
                return NULL; //leaf now a null pointer
            }
            //one subtree
            else if(!c->left && c->right)
            {
                struct Cell *tmp = c->right; //add reference to right subtree root, to ensure it is not deleted
                delete c; //delete the current node
                return tmp; //the current node is now it's right child
            }
            else if(c->left && !c->right)
            {
                struct Cell *tmp = c->left;
                delete c;
                return tmp;
            }
            //two subtrees
            else
            {
                struct Cell *tmp = findMin(c->right); //inorder successor
                c->signal_counter = tmp->signal_counter;
                c->alpha = 1;
                c->duplicateMap.clear();

                for(auto iter = tmp->duplicateMap.begin(); iter != tmp->duplicateMap.end(); ++iter)
                {
                    c->duplicateMap.insert(*iter, tmp->signal_counter);
                }

                c->right = remove(tmp->signal_counter, *tmp->duplicateMap.begin(), c->right, true);
            }

        }

        return c;
    }


    double TumbleTree::removeMin()
    {
        Cell* tmp = findMin(root);

        if(tmp->duplicateMap.numValues() > 1){
            tmp->duplicateMap.erase(*tmp->duplicateMap.begin());
            return tmp->signal_counter;
        }
        Cell* tmp2 = tmp->parent;
        double sc = tmp->signal_counter;
        delete tmp;
        tmp2->left = NULL;
        return sc;

    }

    double TumbleTree::removeMax()
    {
        Cell* tmp = findMax(root);

        if(tmp->duplicateMap.numValues() > 1){
            tmp->duplicateMap.erase(*tmp->duplicateMap.begin());
            return tmp->signal_counter;
        }

        double sc = tmp->signal_counter;
        if(tmp->parent){
            Cell* tmp2 = tmp->parent;
            delete tmp;
            tmp2->left = NULL;
        }
        else
        {
            delete tmp;
        }


        return sc;

    }

    /**
     * finds the cell with the minimum sc
     * @param c starting cell
     * @return cell with min sc
     */
    Cell* TumbleTree::findMin(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else{

            c->signal_counter *=  c->alpha;
            if(c->left != NULL) c->left->alpha *= c->alpha; //propagate down the left subtree
            if(c->right != NULL) c->right->alpha *= c->alpha; //propagate down the right subtree
            c->alpha = 1;

            if(c->left == NULL)
            {
                return c;
            }
            else
            {
                return findMin(c->left);
            }
        }
    }

    /**
     * finds the cell with the maximum sc
     * @param c starting cell
     * @return cell with max sc
     */
    Cell* TumbleTree::findMax(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else
        {
            c->signal_counter *=  c->alpha;
            if(c->left != NULL) c->left->alpha *= c->alpha; //propagate down the left subtree
            if(c->right != NULL) c->right->alpha *= c->alpha; //propagate down the right subtree
            c->alpha = 1; //reset alpha

            if(c->right == NULL)
            {
                return c;
            }
            else
            {
                return findMax(c->right);
            }
        }
    }

    Cell* TumbleTree::find(double sc, VertexHandle vH, Cell* c, double alpha)
    {
        if(c == NULL)
        {
            return NULL;
        }
        else if(sc < c->signal_counter * (alpha * c->alpha))
        {
            return find(sc, vH, c->left, alpha * c->alpha);
        }
        else if(sc > c->signal_counter * (alpha * c->alpha))
        {
            return find(sc, vH, c->right, alpha * c->alpha);
        }
        else
        {
            if(c->duplicateMap.containsKey(vH))
                return c;
            else
            {
                //cout << "Cell found, though it doesnt contain the found handle: " << vH.idx() << endl;
                //cout << "It contains: " << endl;
                for(auto iter = c->duplicateMap.begin(); iter != c->duplicateMap.end(); ++iter)
                {
                    //cout << *iter << " ";
                }
                //cout << endl;
                return NULL; //if the key does not exist in the cell with the suitable signal counter
            }
        }

    }

    void TumbleTree::inorder(Cell* c)
    {
        if(c == NULL)
            return;
        //if(c->left)c->left->alpha *= c->alpha;

        inorder(c->left);

        cout << " | [";
        cout << c->signal_counter;// * c->alpha;
        cout << "{ ";
        for(auto iter = c->duplicateMap.begin(); iter != c->duplicateMap.end(); ++iter)
        {
            cout << *iter << ", ";
        }
        cout << "}";
        cout << "((" << c->alpha << "))";
        cout << "]";

        //if(c->right)c->right->alpha *= c->alpha;

        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(double alpha)
    {
        if(root) root->alpha *= 1;//alpha;
        else cout << "shut up mf " << endl;
    }

    int TumbleTree::size(Cell* c)
    {
        if(c == NULL) return 0;
        return (int)c->duplicateMap.numValues() + size(c->left) + size(c->right);
    }



    TumbleTree::TumbleTree()
    {
        root = NULL;
    }

    TumbleTree::~TumbleTree()
    {
        root = makeEmpty(root);
    }

    // public       ||
    // calls        \/


    // we only need functionality to remove a specific cell.
    double TumbleTree::remove(Cell* c, VertexHandle vH)
    {
        //TODO: FIX PROBLEM FOR SC UPDATES.
        //cout << "#####start remove" <<endl;
        double sc = c->signal_counter;//* c->alpha;
        /*double tmp_sc = sc * c->alpha;
        Cell* tmp = c;
        while(tmp != root) //iterate up the tree to find the correct sc.
        {
            if(!tmp->parent){
                cout << "NO PARENT!!!" << endl;
                break;
            }

            if(tmp->parent->parent == tmp || tmp->parent == tmp) //problem here
            {
                cout << "circlleeeeee" << endl;
                break;
            }
            sc *= tmp->alpha;
            tmp = tmp->parent;
            //cout << "Iterating the parents.. " << endl;
        }
        sc *= tmp->alpha; //include root alpha

        if(c->parent != NULL)
            c->parent->right == c ? c->parent->right = remove(tmp_sc, vH, c) : c->parent->left = remove(tmp_sc, vH, c);
        else if(c == root){
            root = remove(tmp_sc, vH, root);
            cout << "root remove" << endl;
        }
        else{
            cout << "not possible" << endl;
            exit(1);
        }*/


        //cout << "#####end remove" << endl;
        root = remove(sc * c->alpha, vH, root);
        return sc; //return the correct sc
    }

    void TumbleTree::display()
    {
        inorder(root);
        cout << endl;
    }


    Cell* TumbleTree::min()
    {
        return this->findMin(root);
    }

    Cell* TumbleTree::max()
    {
        return this->findMax(root);
    }

    Cell* TumbleTree::find(double sc, VertexHandle vH)
    {
        return find(sc, vH, root);
    }

    void TumbleTree::updateSC(double alpha) {
        this->update(alpha);
    }

    int TumbleTree::size(){
        if(root == NULL) return 0;
        return size(root);
    }

    int TumbleTree::maxDepth(Cell* cell)
    {
        if(cell == NULL)
        {
            return 0;
        }
        else
        {
            return std::max(maxDepth(cell->left), maxDepth(cell->right)) +1;
        }
    }

    int TumbleTree::maxDepth()
    {
        return maxDepth(root);
    }

    int TumbleTree::minDepth(Cell* cell)
    {
        if(cell == NULL)
        {
            return 0;
        }
        else
        {
            return std::min(maxDepth(cell->left), maxDepth(cell->right)) +1;
        }
    }

    int TumbleTree::minDepth()
    {
        return minDepth(root);
    }

    int TumbleTree::sumDepth(Cell *c, int currentDepth)
    {
        if(c == NULL) return 0;
        if(c->right == NULL && c->left == NULL) return currentDepth;
        return sumDepth(c->right, currentDepth + 1) + sumDepth(c->left, currentDepth + 1);
    }

    int TumbleTree::numLeafes(Cell *c)
    {
        if(c == NULL) return 0;
        if(c->right == NULL && c->left == NULL) return 1;
        return numLeafes(c->right) + numLeafes(c->left);
    }

    void TumbleTree::balance()
    {
        vector<Cell*> cells;
        getCellsAsVector(root, cells);
        root = buildTree(cells, 0, (int)cells.size() - 1);
    }

    void TumbleTree::getCellsAsVector(Cell* c, vector<Cell*>& cells)
    {
        if(c == NULL)
            return;
        c->signal_counter *= c->alpha; //update sc and propagate alphas
        if(c->left) c->left->alpha *= c->alpha;
        if(c->right) c->right->alpha *= c->alpha;
        c->alpha = 1;

        getCellsAsVector(c->left, cells);
        cells.push_back(c);
        getCellsAsVector(c->right, cells);
    }

    Cell* TumbleTree::buildTree(std::vector<Cell*>& cells, int start, int end)
    {
        // base case
        if (start > end)
            return NULL;

        /* Get the middle element and make it root */
        int mid = (start + end) / 2;
        Cell* cell = cells[mid];

        /* Using index in Inorder traversal, construct
           left and right subtress */
        cell->left = buildTree(cells, start, mid - 1);
        cell->right = buildTree(cells, mid + 1, end);

        if(cell->left)cell->left->parent = cell;
        if(cell->right)cell->right->parent = cell;

        return cell;
    }



} // namespace lvr2