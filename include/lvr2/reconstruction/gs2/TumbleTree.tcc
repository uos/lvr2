//
// Created by patrick on 4/4/19.
//


namespace lvr2{

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

    Cell* TumbleTree::insert(float sc, Index vH, Cell* c)
    {
        if(c == NULL)
        {
            c = new Cell;
            c->signal_counter = sc;
            c->vH = vH;
            c->left = c->right = NULL;
        }
        else if(sc <= c->signal_counter)
        {
            if(sc == c->signal_counter && vH == c->vH)
            {
                std::cout << "Attempted to insert an existing Cell into the Tumble-Tree" << std::endl;
                return NULL;
            }
            c->left = insert(sc, vH, c->left);
        }
        else if(sc > c->signal_counter)
        {
            c->right = insert(sc, vH, c->right);
        }

        return c; //return the inserted Cell
    }

    //private
    Cell* TumbleTree::remove(float sc, Index vH, Cell* c)
    {
        Cell* tmp;
        if(c == NULL)
        {
            return NULL;
        }
        else if(sc <= c->signal_counter)
        {
            remove(sc, vH, c->left);
        }
        else if(sc > c->signal_counter) {
            remove(sc, vH, c->right);
        }
        else if(vH != c->vH){
            if(findMax(c->left)->signal_counter == sc){ //if there is another equal sc in the left subtree
                remove(sc, vH, c->left); //if the handle index is not correct, we can't delete it...looking for another matching sc
            }
            else return NULL; //not found
        }
        else if(c->left && c->right){
            tmp = findMin(c->right); //find minimum sc cell of the right subtree (should be bigger than all left cells sc's)
            c->signal_counter = tmp->signal_counter; //copy
            c->vH = tmp->vH; //copy
            delete tmp; //delete the minimum of the right subtree, as it now serves as the new subtree-root
        }
        else{
            tmp = c;
            //now we only got one or no subtree left
            if(c->left == NULL)
            {
                c = c->right;
            }
            else if(c->right == NULL)
            {
                c = c->left;
            }
            delete tmp;
        }
    }

    /**
     * finds the cell with the minimum sc, might have a left subtree containing more of it, with equal sc's.
     * @param c starting cell
     * @return cell with min sc
     */
    Cell* TumbleTree::findMin(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else if(c->left == NULL)
            return c;
        else
            return findMin(c->left);
    }

    /**
     * finds the cell with the maximum sc, might have a left subtree containing more of it, with equal sc's.
     * @param c starting cell
     * @return cell with max sc
     */
    Cell* TumbleTree::findMax(Cell* c)
    {
        if(c == NULL)
            return NULL;
        else if(c->right == NULL)
            return c;
        else
            return findMax(c->right);
    }

    void TumbleTree::inorder(Cell* c)
    {
        if(c == NULL)
            return;
        inorder(c->left);
        cout << c->signal_counter << " ";
        inorder(c->right);
    }

    //update the SCs (righ now linear -> O(n), later O(log(n))
    void TumbleTree::update(Cell *c, float alpha)
    {
        if(c == NULL) return;
        c->signal_counter -= c->signal_counter*alpha;
        update(c->left,alpha);
        update(c->right, alpha);
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

    Cell* TumbleTree::insert(float sc, VertexHandle vH)
    {
        Cell* ret = insert(sc, vH.idx(), this->root);
        if(root == NULL)
            root = ret;
        return ret;
    }

    // we only need functionality to remove a specific cell.
    void TumbleTree::remove(Cell* c)
    {
        remove(c->signal_counter, c->vH, this->root); //TODO: use c instead of root
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

    void TumbleTree::updateSC(float alpha) {
        this->update(root, alpha);
    }

} // namespace lvr2