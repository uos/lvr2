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

    Cell* TumbleTree::insertIterative(double sc, VertexHandle vH)
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

                tmp->signal_counter *= tmp->alpha;
                if(tmp->left != NULL) tmp->left->alpha *= tmp->alpha;
                if(tmp->right != NULL) tmp->right->alpha *= tmp->alpha;
                tmp->alpha = 1;

                tmp = tmp->right;
            }
        }

        //found
        tmp->duplicateMap.insert(vH,sc);
        //cout << "Inserting a duplicate" << endl;
        return tmp;

    }

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
                c->left = new Cell();
                c->left->left = NULL;
                c->left->right = NULL;
                c->left->alpha = 1;
                c->left->parent = c;
                c->left->signal_counter = sc;
                c->left->duplicateMap.insert(vH, sc);
                return c->left;
            }
        }
        //if the sc is bigger than the cell's sc, go to the right subtree
        else if(sc > c->signal_counter)
        {
            if(c->right)
                return insert(c->right, sc, vH);
            else //insert new cell, if the right subtree is empty
            {
                c->right = new Cell();
                c->right->left = NULL;
                c->right->right = NULL;
                c->right->alpha = 1;
                c->right->parent = c;
                c->right->signal_counter = sc;
                c->right->duplicateMap.insert(vH, sc);
                return c->right;
            }
        }
        else
        {
            c->duplicateMap.insert(vH, sc);
            return c;
        }
    }

    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits iterative??
    Cell* TumbleTree::remove(double sc, VertexHandle vH, Cell* c, bool removeWhole)
    {
        if(c == NULL) return c; //empty tree or not found

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
                c->duplicateMap.erase(vH);
            }
            //no children
            else if(!c->left && !c->right)
            {
                free(c); //delete leaf node
                return NULL; //leaf now a null pointer
            }
            //one subtree
            else if(!c->left && c->right)
            {
                struct Cell *tmp = c->right; //add reference to right subtree root, to ensure it is not deleted
                free(c); //delete the current node
                return tmp; //the current node is now it's right child
            }
            else if(c->left && !c->right)
            {
                struct Cell *tmp = c->left;
                free(c);
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
        if(c->left)c->left->alpha *= c->alpha;

        inorder(c->left);

        cout << " | [";
        cout << c->signal_counter * c->alpha;
        cout << "{ ";
        for(auto iter = c->duplicateMap.begin(); iter != c->duplicateMap.end(); ++iter)
        {
            cout << *iter << ", ";
        }
        cout << "}";
        cout << "]";

        if(c->right)c->right->alpha *= c->alpha;

        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(double alpha)
    {
        if(root) root->alpha *=  alpha;
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

        double sc = c->signal_counter * c->alpha;
        /*double tmp_sc = sc;
        Cell* tmp = c;
        while(tmp != root) //iterate up the tree to find the correct sc.
        {
            if(!tmp->parent){
                cout << "NO PARENT!!!" << endl;
                break;
            }

            if(tmp->parent->parent == tmp) //problem here
            {
                cout << "circlleeeeee" << endl;
                break;
            }
            sc *= tmp->alpha;
            tmp = tmp->parent;
            //cout << "Iterating the parents.. " << endl;
        }
        sc *= tmp->alpha; //include root alpha

        if(c->parent)
            c->parent->right == c ? c->parent->right = remove(tmp_sc, vH, c) : c->parent->left = remove(tmp_sc, vH, c);
        else
            root = remove(tmp_sc, vH, c);*/

        root = remove(sc, vH, root);
        return sc;
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