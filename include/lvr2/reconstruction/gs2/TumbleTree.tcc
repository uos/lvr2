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

    Cell* TumbleTree::insertIterative(float sc, VertexHandle vH)
    {

        Cell* newCell = new Cell;
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

    //TODO: make it work. expected number of cells after the algorithm: runtime*numsplits iterative??
    Cell* TumbleTree::remove(float sc, VertexHandle vH, Cell* c, bool removeWhole, float alpha)
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
            c->left = remove(sc, vH, c->left, removeWhole, alpha);
            if(c->left) c->left->parent = c;
        }
        else if(sc > c->signal_counter) {
            c->right = remove(sc, vH, c->right, removeWhole, alpha);
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
            c->right = remove(tmp->signal_counter, ret, c->right, true, alpha); // no good...
            if(c->right) c->right->parent = c;

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

    Cell* TumbleTree::find(float sc, VertexHandle vH, Cell* c, float alpha)
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

        std::cout << " | [";
        cout << c->signal_counter * c->alpha;
        std::cout << "]";

        if(c->right)c->right->alpha *= c->alpha;

        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(float alpha)
    {
        root->alpha *= 1;
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
    float TumbleTree::remove(Cell* c, VertexHandle vH)
    {
        //TODO: FIX PROBLEM FOR SC UPDATES.

        //cout << "Starting to remove..." << endl;

        float sc = c->signal_counter;
        /*Cell* tmp = c;
        while(tmp != root) //iteratre up the tree to find the correct sc.
        {
            if(!tmp->parent){
                cout << "NO PARENT!!!" << endl;
                break;
            }
            sc *= tmp->alpha;
            tmp = tmp->parent;
        }
        sc *= tmp->alpha; //include root sc*/
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

    Cell* TumbleTree::find(float sc, VertexHandle vH)
    {
        return find(sc, vH, root);
    }

    void TumbleTree::updateSC(float alpha) {
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